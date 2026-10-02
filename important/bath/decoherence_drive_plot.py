import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 11

# =========================================================================
# Grid of (ergotropy_scar - ergotropy_qubit) vs t from decoherence_drive_sweep.py
#   columns: bath rate (dephasing gamma_phi or relaxation gamma_1)
#   rows:    drive strength A, largest at the top
#
#   python decoherence_drive_plot.py --bath dephasing --N 10
#   python decoherence_drive_plot.py --bath relaxation --N 10 --free-y
# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--bath", choices=("dephasing", "relaxation"), default="dephasing")
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--datadir", default="deco_drive_data")
parser.add_argument("--free-y", action="store_true", help="each panel gets its own y range")
parser.add_argument("--show", action="store_true")
cli = parser.parse_args()

files = sorted(glob.glob(os.path.join(cli.datadir, f"{cli.bath}_N{cli.N}", "g*_A*.npz")))
files = [f for f in files if not f.endswith(".tmp.npz")]
assert files, f"no npz files for bath={cli.bath}, N={cli.N} in {cli.datadir}"

cells = {}
for f in files:
    d = np.load(f)
    cells[(float(d["gamma"]), float(d["A"]))] = (d["tlist"], d["erg_scar"] - d["erg_qubit"])

gammas = sorted({g for g, _ in cells})
As = sorted({A for _, A in cells}, reverse=True)     # largest A on the top row

sym = r"\gamma_\phi" if cli.bath == "dephasing" else r"\gamma_1"

fig, axes = plt.subplots(len(As), len(gammas), squeeze=False,
                         sharex=True, sharey=not cli.free_y,
                         figsize=(2.6 * len(gammas), 1.9 * len(As)))

for i, A in enumerate(As):
    for j, g in enumerate(gammas):
        ax = axes[i, j]
        ax.axhline(0, color="k", lw=0.6, ls="--")

        if (g, A) in cells:
            t, diff = cells[(g, A)]
            ax.plot(t, diff, lw=1.2)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(rf"${sym} = {g:g}$")
        if j == 0:
            ax.set_ylabel(rf"$A = {A:g}$")

fig.supxlabel("Time")
fig.suptitle(rf"$\mathcal{{E}}_{{\rm scar}} - \mathcal{{E}}_{{\rm qubit}}$ (per bandwidth),  "
             f"{cli.bath},  N = {cli.N}")
fig.tight_layout()

out = f"deco_drive_grid_{cli.bath}_N{cli.N}.png"
fig.savefig(out, dpi=200)
print(f"saved {out}  ({len(cells)} of {len(As) * len(gammas)} cells)")

if cli.show:
    plt.show()
