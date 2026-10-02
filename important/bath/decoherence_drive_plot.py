import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# Plots every N found in deco_drive_data/<bath>_N*/ on the same grid.
#   columns: bath rate,  rows: drive strength A (largest on top)
# Makes two pdfs:
#   deco_drive_diff_<bath>.pdf      ergotropy_scar - ergotropy_qubit, all N together
#   deco_drive_erg_<bath>_N<N>.pdf  ergotropy_scar (solid) and ergotropy_qubit (dashed), one pdf per N
#
#   python decoherence_drive_plot.py              (dephasing)
#   python decoherence_drive_plot.py relaxation
# =========================================================================

bath = sys.argv[1] if len(sys.argv) > 1 else "dephasing"
lw = 0.7

# load every finished cell: (N, gamma, A) -> data
data = {}
for f in glob.glob(f"deco_drive_data/{bath}_N*/g*_A*.npz"):
    if not f.endswith(".tmp.npz"):
        d = dict(np.load(f))
        data[(int(d["N"]), float(d["gamma"]), float(d["A"]))] = d

assert data, f"no data found for {bath}"

Ns = sorted({N for N, g, A in data})
gammas = sorted({g for N, g, A in data})
As = sorted({A for N, g, A in data}, reverse=True)

sym = r"\gamma_\phi" if bath == "dephasing" else r"\gamma_1"
print(f"{bath}: N = {Ns}, {len(data)} cells")


def plot_diff(ax, d, color, N):
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.plot(d["tlist"], d["erg_scar"] - d["erg_qubit"], color=color, lw=lw, label=f"N={N}")


def plot_erg(ax, d, color, N):
    ax.plot(d["tlist"], d["erg_scar"], color=color, lw=lw, label=f"scar N={N}")
    ax.plot(d["tlist"], d["erg_qubit"], color=color, lw=lw, ls="--", label=f"qubit N={N}")


def make_grid(plot_cell, Ns_here, title, out):
    fig, axes = plt.subplots(len(As), len(gammas), sharex=True, sharey=True, squeeze=False,
                             figsize=(2.6 * len(gammas), 1.9 * len(As)))

    for i, A in enumerate(As):
        for j, g in enumerate(gammas):
            ax = axes[i, j]
            for k, N in enumerate(Ns_here):
                if (N, g, A) in data:
                    plot_cell(ax, data[(N, g, A)], f"C{k}", N)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf"${sym} = {g:g}$")
            if j == 0:
                ax.set_ylabel(rf"$A = {A:g}$")

    # legend from every panel, so it still works if the top-right cell is missing
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    axes[0, -1].legend(handles.values(), handles.keys(), fontsize=7)
    fig.supxlabel("Time")
    fig.suptitle(f"{title},  {bath}")
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


# difference: every N on the same grid
make_grid(plot_diff, Ns, r"$\mathcal{E}_{\rm scar} - \mathcal{E}_{\rm qubit}$ (per bandwidth)",
          f"deco_drive_diff_{bath}.pdf")

# individual ergotropies: one pdf per N
for N in Ns:
    make_grid(plot_erg, [N], rf"$\mathcal{{E}}_{{\rm scar}}$ (solid),  $\mathcal{{E}}_{{\rm qubit}}$ (dashed),  N = {N}",
              f"deco_drive_erg_{bath}_N{N}.pdf")
