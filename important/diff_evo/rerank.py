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
            sv = r.get("S_at_tmax")
            if sv is not None and sv > thermal_cut:
                continue
            keep.append(r)
        print(f"entropy filter S(t_max) <= {thermal_cut} nats: "
              f"{len(keep)}/{len(rows)} survive")
        rows = keep

    rows.sort(key=lambda r: -r[rank_by])
    return rows[:top]


def show(rows, rank_by):
    head = (f"{'#':>3} {rank_by:>12} {'score':>10} {'deph':>10} {'power':>10} "
            f"{'R_scar':>8} {'R_qub':>8} {'t_max':>7} {'S_nats':>7} "
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
              f"{r.get('S_at_tmax', np.nan):>7.3f} "
              f"{r['x']:>9.2e} {r['y']:>9.2e} {r['z']:>9.2e} "
              f"{r['ds']:>6.3f} {r['dd']:>6.3f} {r.get('wd', np.nan):>7.4f}")


EXPORT_COLUMNS = [
    ("rank",        lambda i, r: i),
    ("score",       lambda i, r: r.get("score")),
    ("score_deph",  lambda i, r: r.get("score_deph")),
    ("score_power", lambda i, r: r.get("score_power")),
    ("score_first", lambda i, r: r.get("score_first")),
    ("maxR_scar",   lambda i, r: r.get("maxR_scar")),
    ("maxR_qubit",  lambda i, r: r.get("maxR_qubit")),
    ("maxRdeph_scar",  lambda i, r: r.get("maxRdeph_scar")),
    ("maxRdeph_qubit", lambda i, r: r.get("maxRdeph_qubit")),
    ("tmax_scar",   lambda i, r: r.get("tmax_scar")),
    ("R1_scar",     lambda i, r: r.get("R1_scar")),
    ("t1_scar",     lambda i, r: r.get("t1_scar")),
    ("S_at_tmax",   lambda i, r: r.get("S_at_tmax")),
    ("S_at_t1",     lambda i, r: r.get("S_at_t1")),
    ("deph_fraction_scar", lambda i, r: r.get("deph_fraction_scar")),
    ("x",  lambda i, r: r.get("x")),
    ("y",  lambda i, r: r.get("y")),
    ("z",  lambda i, r: r.get("z")),
    ("ds", lambda i, r: r.get("ds")),
    ("dd", lambda i, r: r.get("dd")),
    ("wd", lambda i, r: r.get("wd")),
    ("wq", lambda i, r: r.get("wq")),
    ("score_std", lambda i, r: r.get("score_std")),
    ("n_seeds",   lambda i, r: len(r.get("seeds", []))),
    ("elapsed_seconds", lambda i, r: r.get("elapsed_seconds")),
]


def _cells(rows):
    names = [n for n, _ in EXPORT_COLUMNS]
    out = []
    for i, r in enumerate(rows, 1):
        out.append([f(i, r) for _, f in EXPORT_COLUMNS])
    return names, out


def export_csv(rows, path):
    """Ranked table as CSV -- opens in Excel, pandas, anything."""
    import csv

    names, cells = _cells(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in cells:
            w.writerow(["" if v is None else v for v in row])
    print(f"wrote {path}")


def export_md(rows, path, rank_by):
    """
    Ranked table as a markdown table, ready to paste into notes or the
    manuscript draft. Narrower than the CSV on purpose -- the columns a
    reader actually needs to judge a candidate.
    """
    cols = [
        ("#",        lambda i, r: str(i)),
        (rank_by,    lambda i, r: f"{r.get(rank_by, float('nan')):+.5f}"),
        ("score",    lambda i, r: f"{r.get('score', float('nan')):+.4f}"),
        ("deph",     lambda i, r: f"{r.get('score_deph', float('nan')):+.4f}"),
        ("power",    lambda i, r: f"{r.get('score_power', float('nan')):+.4f}"),
        ("R_scar",   lambda i, r: f"{r.get('maxR_scar', float('nan')):.4f}"),
        ("R_qubit",  lambda i, r: f"{r.get('maxR_qubit', float('nan')):.4f}"),
        ("t_max",    lambda i, r: f"{r.get('tmax_scar', float('nan')):.2f}"),
        ("S_nats",   lambda i, r: f"{r.get('S_at_tmax', float('nan')):.3f}"),
        ("x",        lambda i, r: f"{r['x']:.2e}"),
        ("y",        lambda i, r: f"{r['y']:.2e}"),
        ("z",        lambda i, r: f"{r['z']:.2e}"),
        ("ds",       lambda i, r: f"{r['ds']:.3f}"),
        ("dd",       lambda i, r: f"{r['dd']:.3f}"),
        ("wd",       lambda i, r: f"{r.get('wd', float('nan')):.4f}"),
    ]

    lines = [
        f"# DE candidates, ranked by `{rank_by}`",
        "",
        "Read `score` together with `R_scar`, `R_qubit` and `S/Smax`, never alone:",
        "a positive `score` with `S/Smax` above ~0.4 is heating, and a positive",
        "`score` with a small `R_qubit` means the benchmark was crippled rather",
        "than the chain improved.",
        "",
        "| " + " | ".join(c for c, _ in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append("| " + " | ".join(f(i, r) for _, f in cols) + " |")

    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


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
            f"{'S_nats':>7} {'P(>0)':>7}")
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
                   metavar="NATS",
                   help="Drop candidates whose S(t_max) exceeds this many NATS "
                        "(absolute, not a ratio). At N=12 a Haar-random state "
                        "sits at 2.47 and the absolute max is ln(21)=3.045, so "
                        "0.5 keeps only genuinely coherent peaks.")
    p.add_argument("--confirm", type=int, default=0,
                   help="Re-evaluate the shortlist at this many realizations.")
    p.add_argument("--cache", type=str, default="cache")
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--t-max", type=float, default=200.0)
    p.add_argument("--nt", type=int, default=1601)
    p.add_argument("--wm", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    p.add_argument("--csv", type=str, default=None,
                   help="Write the ranked table as CSV (every column).")
    p.add_argument("--md", type=str, default=None,
                   help="Write the ranked table as a markdown table.")
    p.add_argument("--report", type=str, default=None,
                   help="Shorthand: write <name>.csv, <name>.md and <name>.json.")
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

    csv_path, md_path, json_path = args.csv, args.md, args.out

    if args.report:
        stem = Path(args.report)
        stem.parent.mkdir(parents=True, exist_ok=True)
        csv_path = csv_path or f"{stem}.csv"
        md_path = md_path or f"{stem}.md"
        json_path = json_path or f"{stem}.json"

    print()
    if csv_path:
        export_csv(rows, csv_path)
    if md_path:
        export_md(rows, md_path, args.rank_by)
    if json_path:
        Path(json_path).write_text(json.dumps(
            {"ranked_by": args.rank_by, "thermal_cut": args.thermal_cut,
             "ranked": rows, "confirmed": confirmed}, indent=2, default=float))
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
