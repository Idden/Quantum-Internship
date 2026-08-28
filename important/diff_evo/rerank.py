"""
rerank.py
=========

Offline re-ranking of a finished DE search, from the per-worker JSONL eval
logs. No simulation is re-run for the ranking itself.

Why this exists
---------------
The DE drives on `max_t R_scar - max_t R_qubit`, but that metric can be
maximised by heating: a candidate whose maximum occurs late, at high
half-chain entropy, is a thermalised state, not a charged scar state. Every
evaluation therefore also logged the dephased ergotropy, the first peak,
the charging power and the entropy at the peak. This script lets you
re-sort the whole search on any of them after the fact.

It also does the statistics step the search itself cannot afford. With
only a few realizations per objective call the score is inside the
disorder noise, so the DE optimum is partly whichever point won the
disorder lottery. `--confirm K` re-evaluates the top candidates at K
realizations and reports a bootstrap confidence interval over seeds, which
is what the manuscript needs to quote.

Usage
-----
    # rank everything the search ever evaluated, several ways
    python rerank.py --run de_results/N12 --top 15

    # re-run the 10 best dephased-score candidates at 32 realizations
    python rerank.py --run de_results/N12 --rank-by score_deph \
                     --top 10 --confirm 32 --cache cache
"""

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    "score",          # max R difference -- what DE optimised
    "score_deph",     # dephased ergotropy difference -- the honest version
    "score_power",    # max_t R/t difference -- charging power
    "score_first",    # first-peak height difference
]


def load_evals(run_dir):
    """Read every evals_pid*.jsonl under a run directory (or a tree of them)."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.rglob("evals_pid*.jsonl"))

    if not files:
        raise SystemExit(f"no eval logs under {run_dir}")

    records, bad = [], 0
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1        # a line half-written when a job was killed

    print(f"read {len(records)} evaluations from {len(files)} log files"
          + (f" ({bad} truncated lines skipped)" if bad else ""))
    return records


def dedupe(records):
    """DE re-proposes points; keep one row per evaluation key."""
    seen = {}
    for r in records:
        seen[r.get("key", id(r))] = r
    return list(seen.values())


def table(records, rank_by, top, thermal_cut=None):
    rows = [r for r in records if rank_by in r and np.isfinite(r[rank_by])]

    if thermal_cut is not None:
        keep = []
        for r in rows:
            smax = r.get("S_max")
            s = r.get("S_at_tmax")
            if smax and s is not None and s / smax > thermal_cut:
                continue
            keep.append(r)
        print(f"entropy filter S(t_max)/S_max <= {thermal_cut}: "
              f"{len(keep)}/{len(rows)} survive")
        rows = keep

    rows.sort(key=lambda r: -r[rank_by])
    return rows[:top]


def show(rows, rank_by):
    head = (f"{'#':>3} {rank_by:>12} {'score':>10} {'deph':>10} {'power':>10} "
            f"{'R_scar':>8} {'R_qub':>8} {'t_max':>7} {'S/Smax':>7} "
            f"{'x':>9} {'y':>9} {'z':>9} {'ds':>6} {'dd':>6} {'wd':>7}")
    print("\n" + head)
    print("-" * len(head))

    for i, r in enumerate(rows, 1):
        smax = r.get("S_max") or 1.0
        print(f"{i:>3} {r[rank_by]:>+12.5f} {r.get('score', np.nan):>+10.4f} "
              f"{r.get('score_deph', np.nan):>+10.4f} "
              f"{r.get('score_power', np.nan):>+10.4f} "
              f"{r.get('maxR_scar', np.nan):>8.4f} {r.get('maxR_qubit', np.nan):>8.4f} "
              f"{r.get('tmax_scar', np.nan):>7.2f} "
              f"{r.get('S_at_tmax', np.nan) / smax:>7.3f} "
              f"{r['x']:>9.2e} {r['y']:>9.2e} {r['z']:>9.2e} "
              f"{r['ds']:>6.3f} {r['dd']:>6.3f} {r.get('wd', np.nan):>7.4f}")


def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def confirm(rows, n_reals, cache, N, t_max, nt, wm):
    """
    Re-evaluate the shortlist at many realizations, with a bootstrap CI over
    seeds. This is the number to quote: a DE optimum found at 4 realizations
    is not evidence on its own.
    """
    import main as m
    import build_cache as bc

    seeds = list(range(n_reals))
    struct, subspace = bc.load_struct(N, cache)
    fields = bc.load_seeds(N, cache, seeds)

    m.CFG.update({
        "N": N, "struct": struct, "subspace": subspace,
        "seed_fields": fields, "seeds": seeds,
        "tlist": np.linspace(0.0, t_max, nt),
        "wm": wm, "log_dir": None, "verbose": False,
    })

    print(f"\nre-evaluating {len(rows)} candidates at {n_reals} realizations")
    head = (f"{'#':>3} {'score':>10} {'95% CI':>22} {'deph':>10} {'power':>10} "
            f"{'S/Smax':>7} {'P(>0)':>7}")
    print(head)
    print("-" * len(head))

    out = []
    for i, r in enumerate(rows, 1):
        res = m.evaluate_point(r["x"], r["y"], r["z"], r["ds"], r["dd"],
                               r.get("wd", 0.6366896896896898), r.get("wq", 1.0))
        ss = np.array(res["seed_scores"])
        lo, hi = bootstrap_ci(ss)
        frac = float((ss > 0).mean())

        print(f"{i:>3} {res['score']:>+10.5f} [{lo:>+9.5f},{hi:>+9.5f}] "
              f"{res['score_deph']:>+10.5f} {res['score_power']:>+10.5f} "
              f"{res['S_at_tmax'] / res['S_max']:>7.3f} {frac:>7.2f}")

        out.append({**{k: r[k] for k in ("x", "y", "z", "ds", "dd")},
                    "wd": r.get("wd"), "wq": r.get("wq"),
                    "n_reals": n_reals, "ci_low": lo, "ci_high": hi,
                    "frac_seeds_positive": frac,
                    **{k: v for k, v in res.items() if not k.startswith("_")}})
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Re-rank a finished DE search offline.")
    p.add_argument("--run", type=str, required=True,
                   help="Run directory, e.g. de_results/N12 (searched recursively).")
    p.add_argument("--rank-by", type=str, default="score", choices=METRICS)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--thermal-cut", type=float, default=None,
                   help="Drop candidates whose S(t_max)/S_max exceeds this. "
                        "0.15 keeps only genuinely coherent peaks.")
    p.add_argument("--confirm", type=int, default=0,
                   help="Re-evaluate the shortlist at this many realizations.")
    p.add_argument("--cache", type=str, default="cache")
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--t-max", type=float, default=200.0)
    p.add_argument("--nt", type=int, default=1601)
    p.add_argument("--wm", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    records = dedupe(load_evals(args.run))

    for metric in METRICS:
        vals = [r[metric] for r in records if metric in r]
        if vals:
            v = np.array(vals)
            print(f"  {metric:>12}: best {v.max():+.5f}  median {np.median(v):+.5f}  "
                  f"fraction > 0: {(v > 0).mean():.3f}")

    rows = table(records, args.rank_by, args.top, args.thermal_cut)
    show(rows, args.rank_by)

    confirmed = None
    if args.confirm:
        confirmed = confirm(rows, args.confirm, args.cache, args.N,
                            args.t_max, args.nt, args.wm)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ranked": rows, "confirmed": confirmed}, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
