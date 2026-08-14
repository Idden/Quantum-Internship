import glob
import numpy as np

N = 20

for label in ("z", "y", "x"):
    files = sorted(glob.glob(f"xyz_data/parts/{label}_dis_N{N}_task*.npz"))
    parts = [np.load(f) for f in files]

    seeds = np.concatenate([p["seeds"] for p in parts])
    scar = np.concatenate([p["scar"] for p in parts])
    qubit = np.concatenate([p["qubit"] for p in parts])

    order = np.argsort(seeds)   # so the row order doesn't depend on task order
    seeds, scar, qubit = seeds[order], scar[order], qubit[order]

    assert len(np.unique(seeds)) == len(seeds), f"{label}: duplicate seeds"

    np.savez(f"xyz_data/{label}_dis_N{N}_error_bands.npz",
             tlist=parts[0]["tlist"], seeds=seeds, scar=scar, qubit=qubit)

    diff = scar.max(axis=1) - qubit.max(axis=1)
    print(f"{label}: {len(seeds)} reals from {len(files)} tasks   "
          f"diff = {diff.mean():+.4f} +/- {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")