import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARTS = "xyz_amp_data/parts"
OUTDIR = "figures"
BANDS = True          # shade mean +/- std over realizations
FNAME = re.compile(r"([xyz])_dis_N(\d+)_task\d+\.npz$")

# (label, N, xyzdis, ampdis) -> [scar arrays], [qubit arrays]
data = {}
tlist = None

for path in sorted(glob.glob(os.path.join(PARTS, "*.npz"))):
    m = FNAME.search(os.path.basename(path))
    if m is None:
        continue
    label, N = m.group(1), int(m.group(2))

    with np.load(path) as f:
        tlist = f["tlist"]
        for xd, ad, scar, qubit in zip(f["xyzdis"], f["ampdis"], f["scar"], f["qubit"]):
            key = (label, N, round(float(xd), 6), round(float(ad), 6))
            if key not in data:
                data[key] = ([], [])
            data[key][0].append(scar)
            data[key][1].append(qubit)

if not data:
    raise SystemExit(f"no part files found in {PARTS}")

labels = sorted({k[0] for k in data})
Ns = sorted({k[1] for k in data})
xyzdis_list = sorted({k[2] for k in data})
ampdis_list = sorted({k[3] for k in data})

colors = {N: c for N, c in zip(Ns, plt.rcParams["axes.prop_cycle"].by_key()["color"])}

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
                if key not in data:
                    continue

                scar = np.array(data[key][0])
                qubit = np.array(data[key][1])
                c = colors[N]

                ax.plot(tlist, scar.mean(0), color=c, lw=1.2, label=f"N={N} scar")
                ax.plot(tlist, qubit.mean(0), color=c, lw=1.2, ls="--", label=f"N={N} qubit")

                if BANDS and len(scar) > 1:
                    s, q = scar.std(0), qubit.std(0)
                    ax.fill_between(tlist, scar.mean(0) - s, scar.mean(0) + s, color=c, alpha=0.12, lw=0)
                    ax.fill_between(tlist, qubit.mean(0) - q, qubit.mean(0) + q, color=c, alpha=0.08, lw=0)

            if i == 0:
                ax.set_title(f"xyzdis = {xd:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"ampdis = {ad:g}\n" + r"$R(\tau)$", fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel("t", fontsize=9)

    handles, hlabels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, hlabels, loc="upper center", ncol=len(Ns) * 2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"{label}-disorder   (solid = scar, dashed = qubit)", y=1.035, fontsize=12)
    fig.tight_layout()

    out = os.path.join(OUTDIR, f"xyz_grid_{label}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)