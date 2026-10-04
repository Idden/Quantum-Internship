import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# Plots every N found in deco_dis_data/<bath>_<dis>_A<A>_N*/
#   columns: bath rate,  rows: disorder strength W (largest on top)
#   each curve is the mean over the disorder realizations
# Makes:
#   deco_dis_erg_<bath>_<dis>_N<N>.pdf   ergotropy_scar (solid) and ergotropy_qubit (dashed), one pdf per N
#   deco_dis_pop_<bath>_<dis>.pdf        clean scar-tower population, all N together
#
#   python decoherence_disorder_plot.py dephasing amp
#   python decoherence_disorder_plot.py relaxation xyz
# =========================================================================

bath = sys.argv[1] if len(sys.argv) > 1 else "dephasing"
dis = sys.argv[2] if len(sys.argv) > 2 else "amp"
A = 0.1          # must match A in the sweep
lw = 0.7

# load every finished cell: (N, gamma, W) -> data
data = {}
for f in glob.glob(f"deco_dis_data/{bath}_{dis}_A{A:g}_N*/g*_W*.npz"):
    if not f.endswith(".tmp.npz"):
        d = dict(np.load(f))
        data[(int(d["N"]), float(d["gamma"]), float(d["W"]))] = d

assert data, f"no data found for bath={bath}, dis={dis}, A={A}"

Ns = sorted({N for N, g, W in data})
gammas = sorted({g for N, g, W in data})
Ws = sorted({W for N, g, W in data}, reverse=True)

gsym = r"\gamma_\phi" if bath == "dephasing" else r"\gamma_1"
wsym = {"z": r"W_z", "y": r"W_y", "x": r"W_x", "xyz": r"W_{xyz}",
        "amp": r"\delta_A", "freq": r"\delta_\omega"}[dis]
print(f"{bath}, {dis}: N = {Ns}, {len(data)} cells")


def plot_erg(ax, d, color, N):
    ax.plot(d["tlist"], d["erg_scar"].mean(axis=0), color=color, lw=lw, label=f"scar N={N}")
    ax.plot(d["tlist"], d["erg_qubit"].mean(axis=0), color=color, lw=lw, ls="--", label=f"qubit N={N}")


def plot_pop(ax, d, color, N):
    ax.plot(d["tlist"], d["scar_pop"].mean(axis=0), color=color, lw=lw, label=f"N={N}")


def make_grid(plot_cell, Ns_here, title, out):
    fig, axes = plt.subplots(len(Ws), len(gammas), sharex=True, sharey=True, squeeze=False,
                             figsize=(2.6 * len(gammas), 1.9 * len(Ws)))

    for i, W in enumerate(Ws):
        for j, g in enumerate(gammas):
            ax = axes[i, j]
            for k, N in enumerate(Ns_here):
                if (N, g, W) in data:
                    plot_cell(ax, data[(N, g, W)], f"C{k}", N)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf"${gsym} = {g:g}$")
            if j == 0:
                ax.set_ylabel(rf"${wsym} = {W:g}$")

    # legend from every panel, so it still works if the top-right cell is missing
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles[l] = h
    axes[0, -1].legend(handles.values(), handles.keys(), fontsize=7)
    fig.supxlabel("Time")
    fig.suptitle(f"{title},  {bath},  {dis} disorder,  A = {A:g}")
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


# individual ergotropies: one pdf per N
for N in Ns:
    make_grid(plot_erg, [N],
              rf"$\mathcal{{E}}_{{\rm scar}}$ (solid),  $\mathcal{{E}}_{{\rm qubit}}$ (dashed),  N = {N}",
              f"deco_dis_erg_{bath}_{dis}_N{N}.pdf")

# scar-tower population: every N on the same grid
make_grid(plot_pop, Ns, "clean scar-tower population", f"deco_dis_pop_{bath}_{dis}.pdf")
