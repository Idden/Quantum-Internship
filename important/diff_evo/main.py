"""
main.py
=======

Differential evolution search for the disorder + drive parameters that
maximise

    score = mean_seed [ max_t R_scar(t) - max_t R_qubit(t) ]

where R(t) = (<H0>(t) - <H0>(0)) / bandwidth is the normalised ergotropy
of the battery.

What changed from test_main.py
------------------------------
* The numerics moved into scarcore.py, which is qutip-free. qutip is now
  only used by the optional startup cross-check against
  quantumScarFunctions.py, so an API change between qutip 4 and 5 can no
  longer take down a four-hour allocation.
* Everything field-independent is loaded from a disk cache built once by
  build_cache.py, instead of being rebuilt in every worker of every array
  task.
* H0_dis is assembled as a linear combination of cached matrices rather
  than from Python bitstring loops plus fresh RNG draws.
* Every evaluation now also records the DEPHASED ergotropy, the first
  peak, the charging power and the half-chain entropy. DE still drives on
  max R, so the search is comparable with earlier runs, but the eval log
  contains everything needed to re-rank on a better metric offline with
  rerank.py -- no re-running the search.
* `max_t R` is refined with a parabolic correction around the argmax.
  Sampling bias on the old dt = 0.5 grid was worth up to 0.033, the same
  size as the effect being searched for, and it biased the two models
  differently because their frequency content differs.
* --search-wd / --search-wq optionally promote the two drive frequencies
  to search parameters, so neither model is handicapped by sitting off
  resonance.

Why the extra metrics matter
----------------------------
`max_t R` can be maximised by heating. In the earlier metric review one
candidate reached its maximum at t = 101 with half-chain entropy at 30% of
its maximum -- a thermal state, not a scar state -- and only 47% of that
charge survived dephasing. Another reached its maximum at t = 2.1 with
entropy at 5% and 79% surviving. `max_t R` scores those identically.
`R_deph`, `R_1` and `S_vN` tell them apart, which is why every evaluation
now logs all of them.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import scarcore as sc
import build_cache as bc


# ======================================================================
# Process-local context
#
# The old code set module globals in the parent and relied on the DE
# worker pool being FORKED, so the children would inherit them. That is
# not safe: scipy pickles the objective and ships it through the pool's
# task queue, and multiprocessing's start method is not guaranteed to be
# `fork` (it is not on macOS, and newer CPython is moving away from it on
# Linux too). When the assumption breaks the workers silently run with
# default settings -- here that meant `tlist = None` and every evaluation
# dying inside the pool.
#
# So the configuration now travels WITH the objective as plain primitives
# (see class Objective), and the heavy cached arrays are loaded lazily
# once per worker process into _CTX, keyed by (N, cache dir). Forked
# workers still get the parent's copy for free; spawned ones load from
# disk once. Either way it is correct.
# ======================================================================

_CTX = {}

CFG = {
    "N": 12,
    "struct": None,
    "subspace": None,
    "seed_fields": None,      # (v_z, v_y, v_x, v_w), each (n_seeds, N)
    "seeds": [],
    "tlist": None,
    "wd": 0.6366896896896898,
    "wq": 1.0,
    "wm": 1.0,
    "search_wd": False,
    "search_wq": False,
    "log_dir": None,
    "verbose": True,
    "rtol": 1e-8,
    "atol": 1e-10,
}

_MEMO = {}
_MEMO_CAP = 20000
_EVALS_DONE = 0


# ======================================================================
# Parameter vector
# ======================================================================

def de_bounds(args):
    """
    Bounds in the order the objective unpacks them.

    x, y, z are searched in log10 because they span three decades; ds, dd
    and the drive frequencies are searched linearly.
    """
    bounds = [
        (args.logx_min, args.logx_max),
        (args.logy_min, args.logy_max),
        (args.logz_min, args.logz_max),
        (args.ds_min, args.ds_max),
        (args.dd_min, args.dd_max),
    ]
    names = ["log10x", "log10y", "log10z", "ds", "dd"]

    if args.search_wd:
        bounds.append((args.wd_min, args.wd_max))
        names.append("wd")
    if args.search_wq:
        bounds.append((args.wq_min, args.wq_max))
        names.append("wq")

    return bounds, names


def unpack(vec):
    """
    vec = [log10 x, log10 y, log10 z, ds, dd] (+ wd) (+ wq)

    ds is the drive amplitude A; dd is the drive-weight disorder strength.
    wd and wq fall back to the fixed CFG values when not being searched.
    """
    logx, logy, logz, ds, dd = vec[:5]
    k = 5

    wd = float(vec[k]) if CFG["search_wd"] else CFG["wd"]
    k += 1 if CFG["search_wd"] else 0

    wq = float(vec[k]) if CFG["search_wq"] else CFG["wq"]

    return (10.0 ** float(logx), 10.0 ** float(logy), 10.0 ** float(logz),
            float(ds), float(dd), wd, wq)


def eval_key(N, params, seeds, tlist):
    text = "|".join(
        [f"N={N}"]
        + [f"{p:.16e}" for p in params]
        + [f"seeds={list(seeds)}", f"t={tlist[-1]:.6e}", f"nt={len(tlist)}"]
    )
    return hashlib.md5(text.encode()).hexdigest()[:20]


# ======================================================================
# One parameter point
# ======================================================================

def evaluate_point(x, y, z, ds, dd, wd, wq, want_curves=False):
    """
    Run every objective seed at one parameter point and return the score
    plus the full diagnostic metric set.

    Both models see the SAME unit disorder fields for a given seed, so the
    comparison is paired (common random numbers) and the difference of the
    two maxima is far less noisy than either maximum alone.
    """
    struct = CFG["struct"]
    N = CFG["N"]
    tlist = CFG["tlist"]
    Vz, Vy, Vx, Vw = CFG["seed_fields"]

    rows = []
    curves = []

    for k in range(len(CFG["seeds"])):
        v_z, v_y, v_x, v_w = Vz[k], Vy[k], Vx[k], Vw[k]

        # ---- scar chain ------------------------------------------------
        H0, d1 = sc.assemble_scar_H(struct, v_z, v_y, v_x, v_w, x, y, z, dd)
        s = sc.evolve_scar(struct, H0, d1, ds, wd, tlist,
                           rtol=CFG["rtol"], atol=CFG["atol"])
        R_s, Rd_s = sc.scar_metrics(s["E"], s["pops"], s["bandwidth"])

        # ---- decoupled qubits ------------------------------------------
        q = sc.evolve_qubits(N, v_z, v_y, v_x, v_w, x, y, z, dd,
                             ds, wq, CFG["wm"], tlist)
        R_q, Rd_q = sc.qubit_metrics(q["E"], q["pops"], q["bandwidth"])

        # ---- metrics ---------------------------------------------------
        maxR_s, tmax_s = sc.refined_max(R_s, tlist)
        maxR_q, tmax_q = sc.refined_max(R_q, tlist)
        maxRd_s, _ = sc.refined_max(Rd_s, tlist)
        maxRd_q, _ = sc.refined_max(Rd_q, tlist)
        R1_s, t1_s = sc.first_peak(R_s, tlist)
        R1_q, t1_q = sc.first_peak(R_q, tlist)
        P_s, tP_s = sc.max_power(R_s, tlist)
        P_q, _ = sc.max_power(R_q, tlist)

        # entanglement only at the two times that matter, not all nt
        i1 = int(np.argmin(np.abs(tlist - t1_s)))
        imax = int(np.argmin(np.abs(tlist - tmax_s)))
        S1 = sc.half_chain_entropy(struct, s["psi"][:, i1])
        Smax_t = sc.half_chain_entropy(struct, s["psi"][:, imax])

        rows.append({
            "score": maxR_s - maxR_q,
            "score_deph": maxRd_s - maxRd_q,
            "score_power": P_s - P_q,
            "score_first": R1_s - R1_q,
            "maxR_scar": maxR_s, "maxR_qubit": maxR_q,
            "tmax_scar": tmax_s, "tmax_qubit": tmax_q,
            "maxRdeph_scar": maxRd_s, "maxRdeph_qubit": maxRd_q,
            "R1_scar": R1_s, "t1_scar": t1_s,
            "R1_qubit": R1_q, "t1_qubit": t1_q,
            "Pmax_scar": P_s, "t_Pmax_scar": tP_s, "Pmax_qubit": P_q,
            "S_at_t1": S1, "S_at_tmax": Smax_t,
            "deph_fraction_scar": maxRd_s / maxR_s if maxR_s > 1e-12 else 0.0,
        })

        if want_curves:
            curves.append({
                "R_scar": R_s, "R_qubit": R_q,
                "Rdeph_scar": Rd_s, "Rdeph_qubit": Rd_q,
                "psi": s["psi"],
            })

    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["score_std"] = float(np.std([r["score"] for r in rows]))
    out["seed_scores"] = [float(r["score"]) for r in rows]
    out["S_max"] = float((N / 2) * np.log(2.0))

    if want_curves:
        out["_curves"] = curves

    return out


# ======================================================================
# Objective
# ======================================================================

def log_eval(record):
    """
    One append-only JSONL per worker pid. No lock, no cross-process
    interleaving, and re-ranking later is a single pass over these files.
    Thousands of tiny npz files would be a metadata problem on a parallel
    filesystem; one line per evaluation is not.
    """
    if CFG["log_dir"] is None:
        return
    path = Path(CFG["log_dir"]) / f"evals_pid{os.getpid()}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


class Objective:
    """
    The DE objective, carrying its own configuration.

    Only primitives are stored on the instance, so pickling it into a
    worker is cheap and does not depend on how the pool was started. The
    cached matrices are attached to the process on first call.
    """

    def __init__(self, N, cache, seeds, t_max, nt, wd, wq, wm,
                 search_wd, search_wq, log_dir, verbose, rtol, atol):
        self.N = int(N)
        self.cache = str(cache)
        self.seeds = [int(s) for s in seeds]
        self.t_max = float(t_max)
        self.nt = int(nt)
        self.wd = float(wd)
        self.wq = float(wq)
        self.wm = float(wm)
        self.search_wd = bool(search_wd)
        self.search_wq = bool(search_wq)
        self.log_dir = str(log_dir) if log_dir else None
        self.verbose = bool(verbose)
        self.rtol = float(rtol)
        self.atol = float(atol)

    def install(self):
        """Populate this process's CFG from the instance, loading the cache once."""
        ck = (self.N, self.cache, tuple(self.seeds))

        if ck not in _CTX:
            struct, subspace = bc.load_struct(self.N, self.cache)
            fields = bc.load_seeds(self.N, self.cache, self.seeds)
            _CTX[ck] = (struct, subspace, fields)

        struct, subspace, fields = _CTX[ck]

        CFG.update({
            "N": self.N, "struct": struct, "subspace": subspace,
            "seed_fields": fields, "seeds": self.seeds,
            "tlist": np.linspace(0.0, self.t_max, self.nt),
            "wd": self.wd, "wq": self.wq, "wm": self.wm,
            "search_wd": self.search_wd, "search_wq": self.search_wq,
            "log_dir": self.log_dir, "verbose": self.verbose,
            "rtol": self.rtol, "atol": self.atol,
        })

    def __call__(self, vec):
        self.install()
        return objective(vec)


def objective(vec):
    global _EVALS_DONE

    x, y, z, ds, dd, wd, wq = unpack(vec)
    key = eval_key(CFG["N"], (x, y, z, ds, dd, wd, wq), CFG["seeds"], CFG["tlist"])

    if key in _MEMO:
        return -_MEMO[key]

    t0 = time.perf_counter()

    try:
        res = evaluate_point(x, y, z, ds, dd, wd, wq)
        score = float(res["score"])
        elapsed = time.perf_counter() - t0

        if not np.isfinite(score):
            return 1.0e9

        if len(_MEMO) < _MEMO_CAP:
            _MEMO[key] = score
        _EVALS_DONE += 1

        rec = {"key": key, "N": int(CFG["N"]),
               "x": x, "y": y, "z": z, "ds": ds, "dd": dd, "wd": wd, "wq": wq,
               "seeds": [int(s) for s in CFG["seeds"]],
               "elapsed_seconds": float(elapsed), "pid": os.getpid()}
        rec.update({k: v for k, v in res.items() if not k.startswith("_")})
        log_eval(rec)

        if CFG["verbose"]:
            print(
                f"score={score:+.6e} deph={res['score_deph']:+.4e} "
                f"pow={res['score_power']:+.4e} "
                f"Rs={res['maxR_scar']:.4f}@{res['tmax_scar']:.1f} "
                f"Rq={res['maxR_qubit']:.4f} "
                f"S/Smax={res['S_at_tmax'] / res['S_max']:.2f} "
                f"| x={x:.3e} y={y:.3e} z={z:.3e} ds={ds:.3f} dd={dd:.3f} "
                f"wd={wd:.4f} wq={wq:.4f} elapsed={elapsed:.2f}s",
                flush=True,
            )

        return -score

    except Exception as exc:
        print(f"FAILED x={x:.3e} y={y:.3e} z={z:.3e} ds={ds:.3f} dd={dd:.3f}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

        # A returned 1e9 is meant to survive one bad integrator step, not to
        # keep a whole allocation running against a broken model. If the
        # first evaluation in this process dies, or the failure is
        # structural, stop now.
        if _EVALS_DONE == 0 or isinstance(
            exc, (TypeError, NameError, AttributeError, ImportError, FileNotFoundError)
        ):
            raise

        _EVALS_DONE += 1
        return 1.0e9


# ======================================================================
# Final evaluation
# ======================================================================

def final_evaluation(x, y, z, ds, dd, wd, wq, seeds, outdir_fields):
    """
    Re-run the winning point on the (larger) final seed set, keeping the
    full curves plus the scar-subspace projections that show whether the
    charge was actually scar-mediated.
    """
    saved_seeds, saved_fields = CFG["seeds"], CFG["seed_fields"]
    CFG["seeds"] = list(seeds)
    CFG["seed_fields"] = outdir_fields

    try:
        res = evaluate_point(x, y, z, ds, dd, wd, wq, want_curves=True)
    finally:
        CFG["seeds"], CFG["seed_fields"] = saved_seeds, saved_fields

    curves = res.pop("_curves")
    struct, sub = CFG["struct"], CFG["subspace"]
    tlist = CFG["tlist"]

    R_scar = np.array([c["R_scar"] for c in curves])
    R_qubit = np.array([c["R_qubit"] for c in curves])
    Rd_scar = np.array([c["Rdeph_scar"] for c in curves])
    Rd_qubit = np.array([c["Rdeph_qubit"] for c in curves])

    vn = np.zeros((len(curves), len(tlist)))
    scar_probs = None
    subspace_prob = None

    if sub is not None:
        P = sub["scar_states"]                       # (D, n_scar)
        scar_probs = np.zeros((len(curves), P.shape[1], len(tlist)))
        subspace_prob = np.zeros((len(curves), len(tlist)))

    for k, c in enumerate(curves):
        psi = c["psi"]
        for t in range(len(tlist)):
            vn[k, t] = sc.half_chain_entropy(struct, psi[:, t])
        if sub is not None:
            amps = np.abs(sub["scar_states"].conj().T @ psi) ** 2
            scar_probs[k] = amps
            subspace_prob[k] = amps.sum(axis=0)

    out = {
        "summary": res,
        "R_scar": R_scar, "R_qubit": R_qubit,
        "Rdeph_scar": Rd_scar, "Rdeph_qubit": Rd_qubit,
        "vn_entropy": vn,
        "scar_probs": scar_probs,
        "scar_subspace_prob": subspace_prob,
    }
    return out


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="DE search for parameters where the scar chain's max "
                    "normalised ergotropy beats N decoupled qubits."
    )

    p.add_argument("--N", type=int, default=12, help="Even system size, >= 4.")
    p.add_argument("--cache", type=str, default="cache",
                   help="Directory written by build_cache.py.")
    p.add_argument("--outdir", type=str, default="de_results")

    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--de-seed", type=int, default=0)

    p.add_argument("--objective-reals", type=int, default=4,
                   help="Disorder realizations per objective call. The effect "
                        "sits inside the single-realization noise, so 1 is a "
                        "lottery; 3-4 is the working range.")
    p.add_argument("--final-reals", type=int, default=32)
    p.add_argument("--seed-offset", type=int, default=0)

    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=4)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--polish", action="store_true")

    p.add_argument("--logx-min", type=float, default=-3.0)
    p.add_argument("--logx-max", type=float, default=0.0)
    p.add_argument("--logy-min", type=float, default=-3.0)
    p.add_argument("--logy-max", type=float, default=0.0)
    p.add_argument("--logz-min", type=float, default=-3.0)
    p.add_argument("--logz-max", type=float, default=0.0)
    p.add_argument("--ds-min", type=float, default=0.1)
    p.add_argument("--ds-max", type=float, default=5.0)
    p.add_argument("--dd-min", type=float, default=0.01)
    p.add_argument("--dd-max", type=float, default=5.0)

    # Drive frequencies. Off by default: 5 parameters, wd pinned to the
    # clean scar gap, which is the story the manuscript currently tells.
    p.add_argument("--search-wd", action="store_true",
                   help="Promote the scar drive frequency to a search "
                        "parameter. Disorder shifts the effective gap, so the "
                        "clean-chain value is off resonance at large x/y/z.")
    p.add_argument("--search-wq", action="store_true",
                   help="Promote the qubit drive frequency too, so neither "
                        "model is handicapped by being off resonance.")
    p.add_argument("--wd-min", type=float, default=0.3)
    p.add_argument("--wd-max", type=float, default=1.2)
    p.add_argument("--wq-min", type=float, default=0.3)
    p.add_argument("--wq-max", type=float, default=2.0)

    p.add_argument("--wd", type=float, default=0.6366896896896898,
                   help="Fixed scar drive frequency when --search-wd is off. "
                        "The measured clean scar gap is 0.635555 and is "
                        "N-independent to six digits over N = 8..16.")
    p.add_argument("--wq", type=float, default=1.0,
                   help="Fixed qubit drive frequency when --search-wq is off.")
    p.add_argument("--wm", type=float, default=1.0, help="Qubit bare frequency.")

    p.add_argument("--t-max", type=float, default=200.0)
    p.add_argument("--nt", type=int, default=1601,
                   help="Time points on [0, t_max]. The old 400 gave dt = 0.5, "
                        "which biased max R by up to 0.033 -- the size of the "
                        "effect. Output points are cheap; the integrator's own "
                        "step control does not depend on this.")

    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)

    p.add_argument("--quiet-evals", action="store_true")
    p.add_argument("--skip-self-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    N = int(args.N)

    slurm = "SLURM_ARRAY_TASK_ID" in os.environ
    if slurm:
        island = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        label, mode = f"island_{island}", "slurm"
    else:
        island = int(args.de_seed)
        ncpu = max(1, int(args.workers))
        label, mode = f"local_seed_{island}", "local"

    run_dir = Path(args.outdir) / f"N{N}" / label
    evals_dir = run_dir / "evals"
    final_dir = run_dir / "final"
    evals_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    obj_seeds = list(range(args.seed_offset, args.seed_offset + args.objective_reals))
    fin_seeds = list(range(args.seed_offset, args.seed_offset + args.final_reals))

    # ---- build the objective; it carries its own config into the workers
    obj = Objective(
        N=N, cache=args.cache, seeds=obj_seeds,
        t_max=args.t_max, nt=args.nt,
        wd=args.wd, wq=args.wq, wm=args.wm,
        search_wd=args.search_wd, search_wq=args.search_wq,
        log_dir=str(evals_dir), verbose=not args.quiet_evals,
        rtol=args.rtol, atol=args.atol,
    )

    t0 = time.perf_counter()
    obj.install()                       # loads the cache in the parent too
    struct, subspace = CFG["struct"], CFG["subspace"]
    fin_fields = bc.load_seeds(N, args.cache, fin_seeds)
    t_load = time.perf_counter() - t0

    bounds, names = de_bounds(args)

    print(f"mode={mode} island={island} workers={ncpu}", flush=True)
    print(f"N={N} D={struct['D']} (cache loaded in {t_load:.2f}s)", flush=True)
    print(f"objective seeds={obj_seeds}  final seeds={fin_seeds}", flush=True)
    print(f"searching {len(bounds)} parameters: {names}", flush=True)
    print(f"bounds={bounds}", flush=True)
    print(f"wd={'searched' if args.search_wd else args.wd}  "
          f"wq={'searched' if args.search_wq else args.wq}  wm={args.wm}", flush=True)
    print(f"time grid: {args.nt} points on [0, {args.t_max}] "
          f"(dt={args.t_max / (args.nt - 1):.4f})", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    if not args.skip_self_check:
        t0 = time.perf_counter()
        import validate_core
        validate_core.check_hamiltonians(min(N, 10))
        validate_core.check_rng_stream(N)
        validate_core.check_against_qutip(min(N, 10))
        print(f"self-check passed in {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    de = differential_evolution(
        obj, bounds=bounds,
        maxiter=args.maxiter, popsize=args.popsize, tol=args.tol,
        seed=island, workers=ncpu, updating="deferred",
        polish=args.polish, disp=True,
    )
    de_elapsed = time.perf_counter() - t0

    x, y, z, ds, dd, wd, wq = unpack(de.x)
    print(f"\nbest objective score = {-de.fun:+.10e}", flush=True)
    print(f"x={x:.10e} y={y:.10e} z={z:.10e} ds={ds:.10f} dd={dd:.10f} "
          f"wd={wd:.10f} wq={wq:.10f}", flush=True)

    print(f"\nfinal evaluation on {len(fin_seeds)} realizations...", flush=True)
    fin = final_evaluation(x, y, z, ds, dd, wd, wq, fin_seeds, fin_fields)
    s = fin["summary"]

    tag = f"island{island}_N{N}"
    npz_path = final_dir / f"{tag}_result.npz"
    json_path = final_dir / f"{tag}_summary.json"

    payload = {
        "tlist": CFG["tlist"], "N": N, "D": struct["D"],
        "x": x, "y": y, "z": z, "ds": ds, "dd": dd,
        "wd": wd, "wq": wq, "wm": CFG["wm"],
        "objective_seeds": np.array(obj_seeds), "final_seeds": np.array(fin_seeds),
        "R_scar": fin["R_scar"], "R_qubit": fin["R_qubit"],
        "Rdeph_scar": fin["Rdeph_scar"], "Rdeph_qubit": fin["Rdeph_qubit"],
        "vn_entropy": fin["vn_entropy"],
        "de_elapsed_seconds": de_elapsed,
    }
    if fin["scar_probs"] is not None:
        payload.update({
            "scar_probs": fin["scar_probs"],
            "scar_subspace_prob": fin["scar_subspace_prob"],
            "scar_indices": subspace["scar_indices"],
            "scar_energies": subspace["scar_energies"],
            "z2_overlap_zero_scar": subspace["z2_overlap_zero_scar"],
        })
    payload.update({k: v for k, v in s.items() if not isinstance(v, list)})
    payload["seed_scores"] = np.array(s["seed_scores"])

    tmp = npz_path.with_name(f"{npz_path.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, npz_path)

    summary = {
        "run_mode": mode, "island": island, "N": N, "D": int(struct["D"]),
        "searched_parameters": names,
        "x": x, "y": y, "z": z, "ds": ds, "dd": dd, "wd": wd, "wq": wq,
        "objective_score": float(-de.fun),
        "objective_seeds": obj_seeds, "final_seeds": fin_seeds,
        "maxiter": args.maxiter, "popsize": args.popsize, "tol": args.tol,
        "t_max": args.t_max, "nt": args.nt,
        "de_elapsed_seconds": de_elapsed,
        "npz_path": str(npz_path),
        "final": {k: v for k, v in s.items()},
        "success": bool(s["score"] > 0.0),
        "meaning": "success means mean_seed[max R_scar - max R_qubit] > 0 on the "
                   "final seed set. Check score_deph and S_at_tmax/S_max before "
                   "believing it: a positive score with S/S_max near 0.3 is heating.",
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nfinal score        = {s['score']:+.6e} +- {s['score_std']:.3e}", flush=True)
    print(f"dephased score     = {s['score_deph']:+.6e}", flush=True)
    print(f"power score        = {s['score_power']:+.6e}", flush=True)
    print(f"first-peak score   = {s['score_first']:+.6e}", flush=True)
    print(f"max R scar/qubit   = {s['maxR_scar']:.6f} / {s['maxR_qubit']:.6f}", flush=True)
    print(f"S(t_max)/S_max     = {s['S_at_tmax'] / s['S_max']:.3f} "
          f"(near 0 = coherent, near 1 = thermal)", flush=True)
    print(f"saved {npz_path}", flush=True)
    print(f"saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
