import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N = 12            # must match xyz_parallel.py
dis = 0.3

data = {}
reals = 0

for label in ("z", "y", "x"):
    files = sorted(glob.glob(f"xyz_data/parts/{label}_dis_N{N}_task*.npz"))
    if not files:
        raise SystemExit(
            f"no part files matching xyz_data/parts/{label}_dis_N{N}_task*.npz - "
            f"check that N here matches the N in xyz_parallel.py"
        )

    parts = [np.load(f) for f in files]

    seeds = np.concatenate([p["seeds"] for p in parts])
    scar = np.concatenate([p["scar"] for p in parts])
    qubit = np.concatenate([p["qubit"] for p in parts])
    scarprob = np.concatenate([p["scarprob"] for p in parts])

    order = np.argsort(seeds)   # so the row order doesn't depend on task order
    seeds, scar, qubit, scarprob = seeds[order], scar[order], qubit[order], scarprob[order]

    assert len(np.unique(seeds)) == len(seeds), f"{label}: duplicate seeds"

    tlist = parts[0]["tlist"]

    # np.savez(f"xyz_data/{label}_dis_N{N}_error_bands.npz",
    #          tlist=tlist, seeds=seeds, scar=scar, qubit=qubit, scarprob=scarprob)

    data[label] = (scar, qubit, scarprob)
    reals = max(reals, len(seeds))

    diff = scar.max(axis=1) - qubit.max(axis=1)
    print(f"{label}: {len(seeds)} reals from {len(files)} tasks   "
          f"diff = {diff.mean():+.4f} +/- {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")

os.makedirs("figures", exist_ok=True)

# on a cluster with no display matplotlib picks Agg on its own, and plt.show()
# would just print a warning. on your laptop it picks a real backend.
HAS_GUI = matplotlib.get_backend().lower() != "agg"


def band(arr):
    arr = np.atleast_2d(arr)
    m = arr.mean(0)
    sem = arr.std(0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(m)
    return m, sem


# -------------------------------
# R(tau) comparison, one panel per disorder axis
# -------------------------------
panels = [("X Disorder", *data["x"][:2]),
          ("Y Disorder", *data["y"][:2]),
          ("Z Disorder", *data["z"][:2])]

fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
fig.suptitle(r"$\sigma_{xyz}$ Disorder Comparisons")

for ax, (title, s, q) in zip(axs, panels):
    for arr, lab in [(s, "Scar"), (q, "Qubit")]:
        m, sem = band(arr)
        line, = ax.plot(tlist, m, label=lab)
        ax.fill_between(tlist, m - sem, m + sem,
                        color=line.get_color(), alpha=0.3, lw=0)
    ax.set_title(title)
    ax.set_ylabel(r"$R(\tau)$")
    ax.set_ylim(0, 1)

axs[0].legend()
axs[2].set_xlabel("Time")

plt.tight_layout()
plt.savefig(f"figures/xyz_N{N}_dis{dis}_reals{reals}.pdf")

# -------------------------------
# scar overlap, all three axes in one figure
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

for label, lab in [("x", "X Disorder"), ("y", "Y Disorder"), ("z", "Z Disorder")]:
    m, sem = band(data[label][2])
    line, = ax.plot(tlist, m, label=lab)
    ax.fill_between(tlist, m - sem, m + sem,
                    color=line.get_color(), alpha=0.3, lw=0)

ax.set_title("Scar Subspace Overlap")
ax.set_xlabel("Time")
ax.set_ylabel(r"$\sum_n |\langle n_{scar} | \psi(t) \rangle|^2$")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.savefig(f"figures/scarprob_N{N}_dis{dis}_reals{reals}.pdf")

if HAS_GUI:
    plt.show()

# -------------------------------
# summary numbers
# -------------------------------
for title, s, q in panels:
    for arr, lab in [(s, "Scar"), (q, "Qubit")]:
        m, sem = band(arr)
        print(f"{title:12s} {lab:5s}  n={arr.shape[0]:4d}  "
              f"peak mean={m.max():.4f}  max sem={sem.max():.5f}  "
              f"max std={arr.std(0, ddof=1).max():.4f}")

for label, lab in [("x", "X Disorder"), ("y", "Y Disorder"), ("z", "Z Disorder")]:
    m, sem = band(data[label][2])
    print(f"{lab:12s} overlap  start={m[0]:.4f}  end={m[-1]:.4f}  min={m.min():.4f}")