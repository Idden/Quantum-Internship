import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "xyz_amp_data"
OUTDIR = "figures"
BANDS = False          # True -> shade mean +/- std over realizations
QUBIT_N = None         # which N's decoupled-qubit curve to show; None -> smallest available

# (label, N, xyzdis, ampdis) -> list of curves, one per realization
scar = {}
qubit = {}
tlist = None

for path in sorted(glob.glob(f"{DATA}/parts_N*/*.npz")):
    N = int(re.search(r"parts_N(\d+)", path).group(1))
    label = os.path.basename(path)[0]

    f = np.load(path)
    tlist = f["tlist"]

    for xd, ad, s, q in zip(f["xyzdis"], f["ampdis"], f["scar"], f["qubit"]):
        key = (label, N, round(float(xd), 6), round(float(ad), 6))
        scar.setdefault(key, []).append(s)
        qubit.setdefault(key, []).append(q)

if not scar:
    raise SystemExit(f"no part files found in {DATA}/parts_N*/")

labels = sorted({k[0] for k in scar})
Ns = sorted({k[1] for k in scar})
xyzdis_list = sorted({k[2] for k in scar})
ampdis_list = sorted({k[3] for k in scar})

qubit_N = Ns[0] if QUBIT_N is None else QUBIT_N
colors = dict(zip(Ns, plt.rcParams["axes.prop_cycle"].by_key()["color"]))


def plot_mean(ax, curves, color, style, name):
    curves = np.array(curves)
    mean = curves.mean(0)
    ax.plot(tlist, mean, color=color, ls=style, lw=1.2, label=name)

    if BANDS and len(curves) > 1:
        std = curves.std(0)
        ax.fill_between(tlist, mean - std, mean + std, color=color, alpha=0.12, lw=0)


os.makedirs(OUTDIR, exist_ok=True)

for label in labels:
    nrow, ncol = len(ampdis_list), len(xyzdis_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)

    for i, ad in enumerate(ampdis_list):
        for j, xd in enumerate(xyzdis_list):
            ax = axes[i][j]

            for N in Ns:
                key = (label, N, xd, ad)
                if key not in scar:
                    continue

                plot_mean(ax, scar[key], colors[N], "-", f"N={N} scar")
                if N == qubit_N:
                    plot_mean(ax, qubit[key], "k", "--", "decoupled qubits")

            if i == 0:
                ax.set_title(f"xyzdis = {xd:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"ampdis = {ad:g}\n" + r"$R(\tau)$", fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel("t", fontsize=9)

    handles, hlabels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, hlabels, loc="upper center", ncol=len(Ns) + 1,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{label}-disorder   (qubit reference: N={qubit_N})", y=1.035, fontsize=12)
    fig.tight_layout()

    out = os.path.join(OUTDIR, f"xyz_grid_{label}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)