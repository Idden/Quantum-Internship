import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
import time
import numpy as np
import qutip as qt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quantumScarFunctions import *
from densityFunctions import *

# sbatch --export=ALL,BATH=dephasing,DIS=amp decoherence_disorder_sweep.sh

# =========================================================================
# Grid sweep: bath rate (columns) x disorder strength W (rows), at fixed drive A.
# Every grid cell is averaged over `reals` disorder realizations. The scar chain
# and the decoupled qubits get the SAME disorder draw in each realization.
#
#   --dis z | y | x | xyz   on-site sigma field disorder (W on that axis / all three)
#   --dis amp               drive-strength disorder, weights 1 + U(-W, W)
#   --dis freq              drive-frequency disorder: site r driven at wd + d_r,
#                           qubit i at wm + d_i, same d ~ U(-W, W) for both
#
#   python decoherence_disorder_sweep.py --bath dephasing --dis amp
#
# Each cell is saved as one npz (all realizations) once all its realizations
# finish; a resubmit skips cells that already have a file.
# Plot with decoherence_disorder_plot.py.
# =========================================================================

N = 6
wm = 1.0
A = 0.1              # fixed drive strength
reals = 25          # disorder realizations per cell (W = 0 rows only need 1)
t_max = 200          # charging time (drive on)
t_idle = 0.0         # idle time after charging (drive off, bath still on)
n_charge = 500
n_idle = 500

# columns: bath rate. which bath it is comes from --bath
gamma_list = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]

# rows: disorder strength W, one list per disorder type
field_list = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]        # --dis z, y, x, xyz
amp_list = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]          # --dis amp
freq_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]      # --dis freq

BATHS = ("dephasing", "relaxation")
DIS_TYPES = ("z", "y", "x", "xyz", "amp", "freq")
OUTDIR = "deco_dis_data"

# extra steps so long-t_max runs don't stop on the step limit
solver_options = {"nsteps": 100000}

# -------------------------------------------------------------------------
# everything that does not depend on the realization, built once per process.
# module level, so it exists in every worker under fork, spawn or forkserver.
# -------------------------------------------------------------------------
tau_r = get_tau_r(N)
wd = 2*np.pi / tau_r

H0_clean, H_evals_clean, H_evecs_clean, psi0, basisList = get_scar_ham(N, diagonalize=True)
H1_clean, _ = get_scar_H1(N, basisList)
H1_sites, _ = get_scar_H1(N, basisList, indv_qubit=True)    # per-site drive, for freq disorder
qH0_clean, qH1_clean, _ = get_qubit_ham(N, wm=wm)


def get_scar_tower(N, H_evals, H_evecs, psi0):
    """The N+1 scar states of the CLEAN chain. Same recipe as
    scar_overlap_from_states: split the spectrum into N+1 equal energy windows
    and take the eigenstate with the largest Z2 overlap in each; the E=0 one
    is replaced by get_zero_scar."""
    overlaps = np.array([abs(psi0.overlap(v))**2 for v in H_evecs])
    edges = np.linspace(H_evals[0] - 0.5, H_evals[-1] + 0.5, N + 2)

    tower = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((H_evals > lo) & (H_evals < hi))[0]
        assert len(idx) > 0, f"empty energy window [{lo:.3f}, {hi:.3f}] at N={N}"
        tower.append(H_evecs[idx[np.argmax(overlaps[idx])]])

    tower[N // 2] = get_zero_scar(N)[0]
    return tower


# population is always measured in the CLEAN scar tower, so with field
# disorder it starts below 1 (the disordered ground state is not exactly the
# clean bottom scar). with amp / freq disorder H0 is clean and it starts at 1.
scar_tower = get_scar_tower(N, H_evals_clean, H_evecs_clean, psi0)
assert scar_population([H_evecs_clean[0].proj()], scar_tower)[0] > 0.999, \
    "clean ground state is not in the scar tower"


def dis_list(dis):
    if dis in ("z", "y", "x", "xyz"):
        return field_list
    return {"amp": amp_list, "freq": freq_list}[dis]


def field_disorder(dis, W):
    """[z, y, x] strengths, the order get_dis_scar_ham / get_qubit_ham expect."""
    return {"z": [W, 0, 0], "y": [0, W, 0], "x": [0, 0, W],
            "xyz": [W, W, W]}.get(dis, [0, 0, 0])


def bath_rates(bath, gamma):
    """(gamma_phi, gamma_1) for the chosen bath."""
    if bath == "dephasing":
        return gamma, 0.0
    if bath == "relaxation":
        return 0.0, gamma
    raise ValueError(f"unknown bath {bath!r}, expected one of {BATHS}")


def cell_dir(bath, dis):
    return os.path.join(OUTDIR, f"{bath}_{dis}_A{A:g}_N{N}")


def cell_path(bath, dis, gamma, W):
    return os.path.join(cell_dir(bath, dis), f"g{gamma:.6g}_W{W:.6g}.npz")


def run_one(job):
    """One disorder realization of one grid cell, scar chain + qubits."""
    bath, dis, gamma, W, k, seed = job
    gamma_phi, gamma_1 = bath_rates(bath, gamma)
    hd = field_disorder(dis, W)
    ds = W if dis == "amp" else 0.0

    # ---- build this realization. same seed for chain and qubits, and the
    # helper functions draw in the same order, so both see the same disorder
    if any(hd):
        H0, H_evals, H_evecs = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=hd,
                                                fixed_seed=seed, diagonalize=True)
    else:    # amp / freq disorder leave H0 clean, so skip the rediagonalization
        H0, H_evals, H_evecs = H0_clean, H_evals_clean, H_evecs_clean
    qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm, ham_disorder=hd, ds_dis=ds, fixed_seed=seed)

    if dis == "amp":
        H1, _ = get_scar_H1(N, basisList, ds_dis=W, fixed_seed=seed)
    elif dis == "freq":
        H1 = H1_sites
    else:
        H1 = H1_clean

    if dis == "freq":
        dw = np.random.default_rng(seed).uniform(-W, W, N)
        args = {"A": A, **{f"wd{r}": wd + dw[r] for r in range(N)}}
        qargs_list = [{"A": A, "omega": wm + dw[i]} for i in range(N)]
    else:
        args = {"A": A, "omega": wd}
        qargs_list = None
    qargs = {"A": A, "omega": wm}

    rho0 = H_evecs[0].proj()
    bandwidth = H_evals[-1] - H_evals[0]

    # ---- scar chain ----
    c_ops = get_scar_c_ops(N, basisList, gamma_phi=gamma_phi, gamma_1=gamma_1)
    tlist, states = evolve_open(H0, H1, c_ops, rho0, args, t_max, t_idle=t_idle,
                                n_charge=n_charge, n_idle=n_idle, options=solver_options)

    energy = np.real(qt.expect(H0, states))
    W_scar = (energy - energy[0]) / bandwidth
    erg_scar = np.array([calculate_ergotropy(r, H0, H_evals) for r in states]) / bandwidth
    scar_pop = scar_population(states, scar_tower)
    del states

    # ---- decoupled qubits ----
    qc_ops = get_qubit_c_ops(gamma_phi=gamma_phi, gamma_1=gamma_1)
    _, rho_list = evolve_qubits_open(qH0_list, qH1_list, qc_ops, qargs, t_max, t_idle=t_idle,
                                     n_charge=n_charge, n_idle=n_idle, args_list=qargs_list,
                                     options=solver_options)
    q = qubit_battery_series(rho_list, qH0_list)

    return (bath, dis, gamma, W), k, {
        "tlist": tlist, "W_scar": W_scar, "erg_scar": erg_scar, "scar_pop": scar_pop,
        "W_qubit": q["W"], "erg_qubit": q["ergotropy"],
    }


def save_cell(cell, results, seeds):
    bath, dis, gamma, W = cell
    out = {key: np.array([results[k][key] for k in range(len(results))])
           for key in results[0] if key != "tlist"}

    # temp name + rename, so a job killed mid-write never leaves a half-written
    # file that a resubmit would then skip
    path = cell_path(*cell)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, tlist=results[0]["tlist"], bath=bath, dis=dis, gamma=gamma, W=W,
             A=A, N=N, wd=wd, wm=wm, t_max=t_max, t_idle=t_idle,
             seeds=np.array(seeds[:len(results)]), **out)
    os.replace(tmp, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bath", choices=BATHS, default="dephasing",
                        help="which rate goes on the horizontal axis of the grid")
    parser.add_argument("--dis", choices=DIS_TYPES, default="amp",
                        help="which disorder goes on the vertical axis of the grid")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute cells that already have an npz")
    cli = parser.parse_args()
    bath, dis = cli.bath, cli.dis

    # off the cluster these fall back to "task 0 of 1", i.e. run everything
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        ntask = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    elif "SLURM_ARRAY_TASK_MAX" in os.environ:
        ntask = (int(os.environ["SLURM_ARRAY_TASK_MAX"])
                 - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0)) + 1)
    else:
        ntask = 1
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # same seeds in every cell, so neighbouring cells differ only by gamma / W
    seeds = [int(s) for s in np.random.SeedSequence(0).generate_state(reals)]

    # each array task owns whole cells. slice BEFORE skipping finished cells,
    # so every task agrees on who owns what
    cells = [(bath, dis, g, W) for W in dis_list(dis) for g in gamma_list]
    mine = cells[task::ntask]
    todo = [c for c in mine if cli.overwrite or not os.path.exists(cell_path(*c))]

    # W = 0 has no disorder, so every realization would be identical
    n_reals = {c: (1 if c[3] == 0 else reals) for c in todo}
    jobs = [(*c, k, seeds[k]) for c in todo for k in range(n_reals[c])]

    os.makedirs(cell_dir(bath, dis), exist_ok=True)
    print(f"N={N}  bath={bath}  dis={dis}  A={A}  reals={reals}  task {task} of {ntask}  "
          f"ncpu={ncpu}  {len(mine)} cells owned, {len(mine) - len(todo)} already done, "
          f"{len(jobs)} realizations to run", flush=True)

    if not jobs:
        raise SystemExit(0)

    t0 = time.perf_counter()
    results = {c: {} for c in todo}
    with ProcessPoolExecutor(max_workers=min(ncpu, len(jobs))) as pool:
        futures = [pool.submit(run_one, j) for j in jobs]
        for f in as_completed(futures):
            cell, k, res = f.result()
            results[cell][k] = res
            if len(results[cell]) == n_reals[cell]:
                save_cell(cell, results.pop(cell), seeds)
                print(f"  done  gamma={cell[2]:<8g} W={cell[3]:<6g} "
                      f"({time.perf_counter() - t0:.0f} s elapsed)", flush=True)

    print(f"{len(jobs)} realizations in {time.perf_counter() - t0:.1f} s", flush=True)
