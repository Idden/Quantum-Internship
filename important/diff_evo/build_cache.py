"""
build_cache.py
==============

Preprocessing step. Run this ONCE per N before submitting the search; the
DE array tasks then load these files instead of recomputing.

Two caches are written:

  cache/struct_N{N}.npz
      Everything field-independent: the constrained basis, the single-site
      flip sparsity pattern, the clean deformed PXP Hamiltonian, and the
      clean scar tower including the E = 0 scar. The zero scar costs a
      sparse LU plus repeated ARPACK shift-invert solves, and the old code
      paid for it in every worker process of every array task.

  cache/seeds_N{N}.npz
      The unit disorder fields v_z, v_y, v_x, v_w for each seed. These are
      what make H0_dis(x, y, z) = H_clean + z*Dz + y*Ay + x*Ax exact, so
      the whole disorder build collapses to four scalar multiplies. They
      are also the object that guarantees common random numbers: the scar
      model and the qubit model read the SAME v's, so a parameter
      comparison is paired rather than being a disorder lottery.

Usage
-----
    python build_cache.py --N 12 --max-seeds 64 --outdir cache

The seed cache is cheap and tiny (4 x N floats per seed), so build more
seeds than you think you need -- re-ranking the top DE candidates at ~32
realizations is the step that turns a noisy score into a defensible one.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import scarcore as sc


def struct_path(outdir, N):
    return Path(outdir) / f"struct_N{N}.npz"


def seeds_path(outdir, N):
    return Path(outdir) / f"seeds_N{N}.npz"


def build_struct_cache(N, outdir, with_subspace=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    struct = sc.build_structure(N)
    t_struct = time.perf_counter() - t0

    H = struct["H_clean"].tocsr()

    payload = {
        "N": np.int64(struct["N"]),
        "D": np.int64(struct["D"]),
        "S": struct["S"],
        "H_data": H.data,
        "H_indices": H.indices,
        "H_indptr": H.indptr,
        "flip_rows": struct["flip_rows"],
        "flip_cols": struct["flip_cols"],
        "flip_site": struct["flip_site"],
        "flip_phase": struct["flip_phase"],
        "z2": struct["z2"],
        "z2_index": np.int64(struct["z2_index"]),
        "d1_base": struct["d1_base"],
        "strings": np.array(struct["strings"]),
    }

    t0 = time.perf_counter()
    s_thermal, s_thermal_std = sc.thermal_entropy_reference(struct)
    payload["S_thermal"] = np.float64(s_thermal)
    payload["S_thermal_std"] = np.float64(s_thermal_std)
    print(f"thermal entropy reference N={N}: S = {s_thermal:.4f} +- {s_thermal_std:.4f} "
          f"(naive (N/2)ln2 = {(N / 2) * np.log(2):.4f}) "
          f"in {time.perf_counter() - t0:.2f}s", flush=True)

    t_sub = 0.0
    if with_subspace:
        t0 = time.perf_counter()
        sub = sc.build_scar_subspace(struct)
        t_sub = time.perf_counter() - t0
        payload.update(
            {
                "scar_states": sub["scar_states"],
                "scar_indices": sub["scar_indices"],
                "scar_energies": sub["scar_energies"],
                "z2_overlap_zero_scar": np.float64(sub["z2_overlap_zero_scar"]),
                "clean_eigenvalues": sub["clean_eigenvalues"],
            }
        )

    path = struct_path(outdir, N)
    np.savez_compressed(path, **payload)

    size_mb = path.stat().st_size / 1e6
    print(
        f"struct N={N}: D={struct['D']} built in {t_struct:.2f}s "
        f"(+ scar subspace {t_sub:.2f}s) -> {path} [{size_mb:.1f} MB]",
        flush=True,
    )
    return path


def build_seed_cache(N, max_seeds, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = np.arange(max_seeds, dtype=int)
    Vz = np.empty((max_seeds, N))
    Vy = np.empty((max_seeds, N))
    Vx = np.empty((max_seeds, N))
    Vw = np.empty((max_seeds, N))

    t0 = time.perf_counter()
    for k, s in enumerate(seeds):
        Vz[k], Vy[k], Vx[k], Vw[k] = sc.draw_unit_fields(N, int(s))
    dt = time.perf_counter() - t0

    path = seeds_path(outdir, N)
    np.savez_compressed(path, seeds=seeds, v_z=Vz, v_y=Vy, v_x=Vx, v_w=Vw)

    print(f"seeds  N={N}: {max_seeds} realizations in {dt:.2f}s -> {path}", flush=True)
    return path


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_struct(N, outdir):
    """
    Load the per-N cache back into the dict shape scarcore expects.

    mmap_mode is deliberately NOT used: the arrays are small and the DE
    workers are forked processes that share these pages copy-on-write,
    which is cheaper than every worker touching the parallel filesystem.
    """
    import scipy.sparse as sp

    path = struct_path(outdir, N)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:  python build_cache.py --N {N} --outdir {outdir}"
        )

    z = np.load(path, allow_pickle=False)
    D = int(z["D"])

    struct = {
        "N": int(z["N"]),
        "D": D,
        "S": z["S"],
        "H_clean": sp.csr_matrix(
            (z["H_data"], z["H_indices"], z["H_indptr"]), shape=(D, D)
        ),
        "flip_rows": z["flip_rows"],
        "flip_cols": z["flip_cols"],
        "flip_site": z["flip_site"],
        "flip_phase": z["flip_phase"],
        "z2": z["z2"],
        "z2_index": int(z["z2_index"]),
        "d1_base": z["d1_base"],
        "strings": [str(s) for s in z["strings"]],
    }

    if "S_thermal" in z.files:
        struct["S_thermal"] = float(z["S_thermal"])

    sub = None
    if "scar_states" in z.files:
        sub = {
            "scar_states": z["scar_states"],
            "scar_indices": z["scar_indices"],
            "scar_energies": z["scar_energies"],
            "z2_overlap_zero_scar": float(z["z2_overlap_zero_scar"]),
            "clean_eigenvalues": z["clean_eigenvalues"],
        }

    return struct, sub


def load_seeds(N, outdir, seeds):
    """Unit disorder fields for the requested seeds, as (n_seeds, N) arrays."""
    path = seeds_path(outdir, N)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:  python build_cache.py --N {N} --outdir {outdir}"
        )

    z = np.load(path, allow_pickle=False)
    available = z["seeds"]
    lookup = {int(s): i for i, s in enumerate(available)}

    missing = [s for s in seeds if int(s) not in lookup]
    if missing:
        raise ValueError(
            f"seed cache for N={N} has {len(available)} seeds but {missing} were "
            f"requested. Rebuild with --max-seeds {max(missing) + 1}."
        )

    rows = [lookup[int(s)] for s in seeds]
    return z["v_z"][rows], z["v_y"][rows], z["v_x"][rows], z["v_w"][rows]


def parse_args():
    p = argparse.ArgumentParser(description="Build the per-N and per-seed caches.")
    p.add_argument("--N", type=int, nargs="+", default=[12],
                   help="One or more even system sizes.")
    p.add_argument("--max-seeds", type=int, default=64,
                   help="Number of disorder realizations to precompute.")
    p.add_argument("--outdir", type=str, default="cache")
    p.add_argument("--no-subspace", action="store_true",
                   help="Skip the clean scar tower (which needs the sparse LU). "
                        "Only do this if you will not run final diagnostics.")
    return p.parse_args()


def main():
    args = parse_args()
    for N in args.N:
        build_struct_cache(N, args.outdir, with_subspace=not args.no_subspace)
        build_seed_cache(N, args.max_seeds, args.outdir)


if __name__ == "__main__":
    main()
