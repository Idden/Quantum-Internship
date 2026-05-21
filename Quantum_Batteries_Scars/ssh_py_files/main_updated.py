import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution

from quantumScarFunctions import *


# ============================================================
# Global defaults
# ============================================================

wd = 0.6366896896896898
wm = 1.0
t_max = 200.0
tlist = np.linspace(0.0, t_max, 400)

DEFAULT_OUTDIR = "/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/data"


# ============================================================
# Differential evolution globals
# These are used because scipy workers need a picklable objective.
# Do not make the objective a nested function.
# ============================================================

_DE_N = None
_DE_SEEDS = None
_DE_CACHE_DIR = None
_DE_LOG_EVERY_EVAL = True


# ============================================================
# Utility functions
# ============================================================

def safe_float_name(x):
    return f"{float(x):.6f}".replace("-", "m").replace(".", "p")


def unpack_de_vector(vec):
    """
    Differential evolution searches over:

        vec[0] = log10(x)
        vec[1] = log10(y)
        vec[2] = log10(z)
        vec[3] = ds
        vec[4] = dd

    This keeps x, y, z logarithmic, like your old np.logspace sweep.
    """
    logx, logy, logz, ds, dd = vec

    x = 10.0 ** float(logx)
    y = 10.0 ** float(logy)
    z = 10.0 ** float(logz)
    ds = float(ds)
    dd = float(dd)

    return x, y, z, ds, dd


def make_eval_hash(N, x, y, z, ds, dd, seeds):
    text = (
        f"N={N}|"
        f"x={x:.16e}|y={y:.16e}|z={z:.16e}|"
        f"ds={ds:.16e}|dd={dd:.16e}|"
        f"seeds={list(seeds)}|"
        f"wd={wd:.16e}|tmax={t_max:.16e}|nt={len(tlist)}"
    )
    return hashlib.md5(text.encode()).hexdigest()[:20]


def atomic_savez(path, **kwargs):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp_path, **kwargs)
    os.replace(tmp_path, path)


# ============================================================
# Scar simulation
# ============================================================

def simulate_scar_rtau(N, x, y, z, ds, dd, seed):
    """
    Simulates the driven disordered scar system and returns Rtau_scar(t).
    """
    np.random.seed(int(seed))

    H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)

    H0_dis, eigenvalues_dis, eigenstates_dis = get_dis_scar_ham(
        H0_clean,
        N,
        basisList,
        ham_disorder=[z, y, x],
    )

    H1, drive_weights = get_scar_H1(
        N,
        basisList,
        ds_dis=dd,
    )

    bandwidth = float(eigenvalues_dis[-1] - eigenvalues_dis[0])

    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"Bad scar bandwidth: {bandwidth}")

    args = {"A": ds, "omega": wd}
    H = qt.QobjEvo([H0_dis, [H1, coeff]], args=args)

    initial_state = eigenstates_dis[0]
    sol = qt.sesolve(H, initial_state, tlist, e_ops=[H0_dis])

    energy = np.array(np.real(sol.expect[0]), dtype=float)
    Rtau_scar = (energy - energy[0]) / bandwidth

    return Rtau_scar


# ============================================================
# Decoupled qubit simulation
# ============================================================

def simulate_decoupled_qubit_rtau(N, x, y, z, ds, dd, seed):
    """
    Simulates the decoupled-qubit comparison model.

    Your new get_qubit_ham returns lists of single-qubit H0_i and H1_i.
    We evolve each qubit independently, sum energy absorbed, then divide by
    total qubit bandwidth.
    """
    np.random.seed(int(seed))

    qH0_list, qH1_list = get_qubit_ham(
        N,
        wm=wm,
        ham_disorder=[z, y, x],
        ds_dis=dd,
    )

    args = {"A": ds, "omega": wm}

    total_delta_energy = np.zeros(len(tlist), dtype=float)
    total_bandwidth = 0.0

    for qH0, qH1 in zip(qH0_list, qH1_list):
        eigenvalues, eigenstates = qH0.eigenstates()
        bandwidth = float(eigenvalues[-1] - eigenvalues[0])

        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError(f"Bad qubit bandwidth: {bandwidth}")

        total_bandwidth += bandwidth

        initial_state = eigenstates[0]
        qH = qt.QobjEvo([qH0, [qH1, coeff]], args=args)
        sol = qt.sesolve(qH, initial_state, tlist, e_ops=[qH0])

        energy = np.array(np.real(sol.expect[0]), dtype=float)
        total_delta_energy += energy - energy[0]

    if total_bandwidth <= 0 or not np.isfinite(total_bandwidth):
        raise ValueError(f"Bad total qubit bandwidth: {total_bandwidth}")

    Rtau_qubit = total_delta_energy / total_bandwidth

    return Rtau_qubit


# ============================================================
# Scoring
# ============================================================

def evaluate_parameters(N, x, y, z, ds, dd, seeds, save_curves=False):
    """
    Score definition:

        score = mean_seed[ max(Rtau_scar) - max(Rtau_decoupled_qubits) ]

    The optimizer maximizes this score.

    score > 0 means scar wins over decoupled qubits.
    """
    seed_scores = []

    scar_curves = []
    qubit_curves = []

    for seed in seeds:
        Rtau_scar = simulate_scar_rtau(N, x, y, z, ds, dd, seed)
        Rtau_qubit = simulate_decoupled_qubit_rtau(N, x, y, z, ds, dd, seed)

        max_scar = float(np.max(Rtau_scar))
        max_qubit = float(np.max(Rtau_qubit))
        seed_score = max_scar - max_qubit

        seed_scores.append(seed_score)

        if save_curves:
            scar_curves.append(Rtau_scar)
            qubit_curves.append(Rtau_qubit)

    seed_scores = np.array(seed_scores, dtype=float)

    result = {
        "score": float(np.mean(seed_scores)),
        "score_std": float(np.std(seed_scores)),
        "seed_scores": seed_scores,
        "max_seed_score": float(np.max(seed_scores)),
        "min_seed_score": float(np.min(seed_scores)),
    }

    if save_curves:
        scar_curves = np.array(scar_curves, dtype=float)
        qubit_curves = np.array(qubit_curves, dtype=float)

        result.update(
            {
                "Rtau_scar_mean": np.mean(scar_curves, axis=0),
                "Rtau_scar_std": np.std(scar_curves, axis=0),
                "Rtau_qubit_mean": np.mean(qubit_curves, axis=0),
                "Rtau_qubit_std": np.std(qubit_curves, axis=0),
                "Rtau_scar_all": scar_curves,
                "Rtau_qubit_all": qubit_curves,
                "max_Rtau_scar_mean": float(np.max(np.mean(scar_curves, axis=0))),
                "max_Rtau_qubit_mean": float(np.max(np.mean(qubit_curves, axis=0))),
            }
        )

    return result


# ============================================================
# Picklable scipy objective
# ============================================================

def differential_evolution_objective(vec):
    global _DE_N, _DE_SEEDS, _DE_CACHE_DIR, _DE_LOG_EVERY_EVAL

    N = int(_DE_N)
    seeds = list(_DE_SEEDS)
    cache_dir = Path(_DE_CACHE_DIR)

    x, y, z, ds, dd = unpack_de_vector(vec)

    eval_key = make_eval_hash(N, x, y, z, ds, dd, seeds)
    cache_path = cache_dir / f"eval_{eval_key}.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                score = float(data["score"])
            return -score
        except Exception:
            pass

    start_time = time.perf_counter()

    try:
        result = evaluate_parameters(
            N=N,
            x=x,
            y=y,
            z=z,
            ds=ds,
            dd=dd,
            seeds=seeds,
            save_curves=False,
        )

        score = float(result["score"])

        if not np.isfinite(score):
            return 1.0e9

        elapsed = time.perf_counter() - start_time

        atomic_savez(
            cache_path,
            N=N,
            x=x,
            y=y,
            z=z,
            ds=ds,
            dd=dd,
            seeds=np.array(seeds, dtype=int),
            score=score,
            score_std=float(result["score_std"]),
            seed_scores=result["seed_scores"],
            elapsed_seconds=float(elapsed),
        )

        if _DE_LOG_EVERY_EVAL:
            print(
                f"score={score:+.8e} "
                f"N={N} "
                f"x={x:.4e} y={y:.4e} z={z:.4e} "
                f"ds={ds:.4e} dd={dd:.4e} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

        # scipy minimizes, so return negative score.
        return -score

    except Exception as exc:
        print(
            f"FAILED x={x:.4e} y={y:.4e} z={z:.4e} ds={ds:.4e} dd={dd:.4e}: {exc}",
            flush=True,
        )
        return 1.0e9


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Use scipy differential evolution to find parameters where max scar Rtau beats max decoupled-qubit Rtau."
    )

    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)

    # Number of disorder seeds used inside the optimizer.
    # Keep this small. This is the expensive part.
    parser.add_argument("--objective-reals", type=int, default=1)

    # Number of seeds used once at the end for the best point.
    parser.add_argument("--final-reals", type=int, default=500)

    # Differential evolution controls.
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--polish", action="store_true")

    # Bounds. x/y/z are log10 bounds.
    parser.add_argument("--logx-min", type=float, default=-3.0)
    parser.add_argument("--logx-max", type=float, default=0.0)
    parser.add_argument("--logy-min", type=float, default=-3.0)
    parser.add_argument("--logy-max", type=float, default=0.0)
    parser.add_argument("--logz-min", type=float, default=-3.0)
    parser.add_argument("--logz-max", type=float, default=0.0)
    parser.add_argument("--ds-min", type=float, default=0.1)
    parser.add_argument("--ds-max", type=float, default=5.0)
    parser.add_argument("--dd-min", type=float, default=0.01)
    parser.add_argument("--dd-max", type=float, default=5.0)

    # If enabled, stop after finding first final result with score > 0.
    # Differential evolution itself still runs to completion; this flag mainly affects final reporting logic.
    parser.add_argument("--quiet-evals", action="store_true")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    global _DE_N, _DE_SEEDS, _DE_CACHE_DIR, _DE_LOG_EVERY_EVAL

    args = parse_args()

    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    outdir = Path(args.outdir)
    run_dir = outdir / "de_scar_beats_qubit" / f"N{args.N}" / f"island_{array_id}"
    cache_dir = run_dir / "cache"
    final_dir = run_dir / "final"

    cache_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    objective_seeds = list(range(args.objective_reals))
    final_seeds = list(range(args.final_reals))

    _DE_N = int(args.N)
    _DE_SEEDS = objective_seeds
    _DE_CACHE_DIR = str(cache_dir)
    _DE_LOG_EVERY_EVAL = not args.quiet_evals

    bounds = [
        (args.logx_min, args.logx_max),
        (args.logy_min, args.logy_max),
        (args.logz_min, args.logz_max),
        (args.ds_min, args.ds_max),
        (args.dd_min, args.dd_max),
    ]

    print("Starting differential evolution search", flush=True)
    print(f"Array island: {array_id}", flush=True)
    print(f"CPUs on this node: {num_cpus}", flush=True)
    print(f"N: {args.N}", flush=True)
    print(f"Objective seeds: {objective_seeds}", flush=True)
    print(f"Final seeds: {final_seeds}", flush=True)
    print(f"Bounds: {bounds}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    start_time = time.perf_counter()

    de_result = differential_evolution(
        differential_evolution_objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        seed=array_id,
        workers=num_cpus,
        updating="deferred",
        polish=args.polish,
        disp=True,
    )

    de_elapsed = time.perf_counter() - start_time

    best_x, best_y, best_z, best_ds, best_dd = unpack_de_vector(de_result.x)
    objective_score = -float(de_result.fun)

    print("\nBest objective result", flush=True)
    print(f"objective_score = {objective_score:+.10e}", flush=True)
    print(f"x = {best_x:.16e}", flush=True)
    print(f"y = {best_y:.16e}", flush=True)
    print(f"z = {best_z:.16e}", flush=True)
    print(f"ds = {best_ds:.16e}", flush=True)
    print(f"dd = {best_dd:.16e}", flush=True)

    print("\nRunning final evaluation on best point...", flush=True)

    final_result = evaluate_parameters(
        N=args.N,
        x=best_x,
        y=best_y,
        z=best_z,
        ds=best_ds,
        dd=best_dd,
        seeds=final_seeds,
        save_curves=True,
    )

    final_score = float(final_result["score"])
    success = bool(final_score > 0.0)

    tag = (
        f"island{array_id}_N{args.N}"
        f"_x{safe_float_name(best_x)}"
        f"_y{safe_float_name(best_y)}"
        f"_z{safe_float_name(best_z)}"
        f"_ds{safe_float_name(best_ds)}"
        f"_dd{safe_float_name(best_dd)}"
    )

    final_npz_path = final_dir / f"{tag}_result.npz"
    summary_path = final_dir / f"{tag}_summary.json"

    atomic_savez(
        final_npz_path,
        tlist=tlist,
        N=int(args.N),
        wd=float(wd),
        t_max=float(t_max),
        x=float(best_x),
        y=float(best_y),
        z=float(best_z),
        ds=float(best_ds),
        dd=float(best_dd),
        objective_seeds=np.array(objective_seeds, dtype=int),
        final_seeds=np.array(final_seeds, dtype=int),
        objective_score=float(objective_score),
        final_score=float(final_score),
        final_score_std=float(final_result["score_std"]),
        seed_scores=final_result["seed_scores"],
        Rtau_scar_mean=final_result["Rtau_scar_mean"],
        Rtau_scar_std=final_result["Rtau_scar_std"],
        Rtau_qubit_mean=final_result["Rtau_qubit_mean"],
        Rtau_qubit_std=final_result["Rtau_qubit_std"],
        Rtau_scar_all=final_result["Rtau_scar_all"],
        Rtau_qubit_all=final_result["Rtau_qubit_all"],
        max_Rtau_scar_mean=float(final_result["max_Rtau_scar_mean"]),
        max_Rtau_qubit_mean=float(final_result["max_Rtau_qubit_mean"]),
        success=np.array(success),
        de_elapsed_seconds=float(de_elapsed),
    )

    summary = {
        "success": success,
        "meaning": "success means final_score = mean_seed[max(Rtau_scar) - max(Rtau_decoupled_qubits)] > 0",
        "array_id": array_id,
        "num_cpus": num_cpus,
        "N": int(args.N),
        "wd": float(wd),
        "t_max": float(t_max),
        "num_t_points": int(len(tlist)),
        "x": float(best_x),
        "y": float(best_y),
        "z": float(best_z),
        "ds": float(best_ds),
        "dd": float(best_dd),
        "objective_score": float(objective_score),
        "final_score": float(final_score),
        "final_score_std": float(final_result["score_std"]),
        "max_Rtau_scar_mean": float(final_result["max_Rtau_scar_mean"]),
        "max_Rtau_qubit_mean": float(final_result["max_Rtau_qubit_mean"]),
        "objective_reals": int(args.objective_reals),
        "final_reals": int(args.final_reals),
        "objective_seeds": objective_seeds,
        "final_seeds": final_seeds,
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "tol": float(args.tol),
        "polish": bool(args.polish),
        "de_elapsed_seconds": float(de_elapsed),
        "npz_path": str(final_npz_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal result", flush=True)
    print(f"success = {success}", flush=True)
    print(f"final_score = {final_score:+.10e}", flush=True)
    print(f"max scar Rtau mean = {final_result['max_Rtau_scar_mean']:.10e}", flush=True)
    print(f"max qubit Rtau mean = {final_result['max_Rtau_qubit_mean']:.10e}", flush=True)
    print(f"Saved npz: {final_npz_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
