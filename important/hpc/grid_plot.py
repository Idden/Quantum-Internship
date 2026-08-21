import os
import re
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARTS = "xyz_amp_data/parts"
OUTDIR = "figures"
BANDS = False          # True -> shade mean +/- std over realizations
QUBIT_N = None         # which N's decoupled-qubit curve to show; None -> smallest available

data = defaultdict(list)   # (label, N, xyzdis, ampdis) -> [(scar, qubit), ...]
tlist = None

for path in sorted(glob.glob(os.path.join(PARTS, "*.npz"))):
    m = re.search(r"([xyz])_dis_N(\d+)_task\d+\.npz$", os.path.basename(path))
    if m is None:
        continue

    with np.load(path) as f:
        tlist = f["tlist"]
        for xd, ad, scar, qubit in zip(f["xyzdis"], f["ampdis"], f["scar"], f["qubit"]):
            key = (m.group(1), int(m.group(2)), round(float(xd), 6), round(float(ad), 6))
            data[key].append((scar, qubit))

if not data:
    raise SystemExit(f"no part files found in {PARTS}")

labels = sorted({k[0] for k in data})
Ns = sorted({k[1] for k in data})
xyzdis_list = sorted({k[2] for k in data})
ampdis_list = sorted({k[3] for k in data})

qubit_N = Ns[0] if QUBIT_N is None else QUBIT_N

colors = dict(zip(Ns, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
os.makedirs(OUTDIR, exist_ok=True)

for label in labels:
    nrow, ncol = len(ampdis_list), len(xyzdis_list)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)

    for i, ad in enumerate(ampdis_list):
        for j, xd in enumerate(xyzdis_list):
            ax = axes[i][j]

            curves = []
            for N in Ns:
                runs = data.get((label, N, xd, ad))
                if not runs:
                    continue

                curves.append((np.array([r[0] for r in runs]), colors[N], "-", f"N={N} scar"))
                if N == qubit_N:
                    curves.append((np.array([r[1] for r in runs]), "k", "--", "decoupled qubits"))

            for curve, c, style, name in curves:
                mean = curve.mean(0)
                ax.plot(tlist, mean, color=c, ls=style, lw=1.2, label=name)
                if BANDS and len(curve) > 1:
                    std = curve.std(0)
                    ax.fill_between(tlist, mean - std, mean + std, color=c, alpha=0.12, lw=0)

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
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)