import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np
import qutip as qt
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quantumScarFunctions import *
from densityFunctions import *

# sbatch --export=ALL,BATH=dephasing decoherence_drive_sweep.sh

# =========================================================================
# Grid sweep: bath rate (columns) x drive strength A (rows), no disorder.
# One grid cell = one exact Lindblad solve for the scar chain + one for the
# decoupled qubits. Each cell is saved as its own npz the moment it finishes,
# so a job that hits walltime keeps what it finished and a resubmit skips it.
#
#   python decoherence_drive_sweep.py --bath dephasing
#   python decoherence_drive_sweep.py --bath relaxation
#
# Plot with decoherence_drive_plot.py.
# =========================================================================

N = 8
wm = 1.0
t_max = 200          # charging time (drive on)
t_idle = 0.0         # idle time after charging (drive off, bath still on)
n_charge = 500
n_idle = 500

# columns: bath rate. which bath it is comes from --bath
gamma_list = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]

# rows: drive strength A. A = 0 is a bath-only control row
A_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

BATHS = ("dephasing", "relaxation")
OUTDIR = "deco_drive_data"

# extra steps so large-A / long-t_max runs don't stop on the step limit
solver_options = {"nsteps": 100000}

# -------------------------------------------------------------------------
# everything that does not depend on (gamma, A), built once per process.
# module level, so it exists in every worker under fork, spawn or forkserver.
# -------------------------------------------------------------------------
tau_r = get_tau_r(N)
wd = 2*np.pi / tau_r

H0, H_evals, H_evecs, psi0, basisList = get_scar_ham(N, diagonalize=True)
H1, _ = get_scar_H1(N, basisList)
rho0 = H_evecs[0].proj()
bandwidth = H_evals[-1] - H_evals[0]

qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm)


def get_scar_tower(N, H_evals, H_evecs, psi0):
    """The N+1 scar states. Same recipe as scar_overlap_from_states: split the
    spectrum into N+1 equal energy windows and take the eigenstate with the
    largest Z2 overlap in each; the E=0 one is replaced by get_zero_scar, since
    it sits inside the degenerate zero-energy manifold."""
    overlaps = np.array([abs(psi0.overlap(v))**2 for v in H_evecs])
    edges = np.linspace(H_evals[0] - 0.5, H_evals[-1] + 0.5, N + 2)

    tower = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((H_evals > lo) & (H_evals < hi))[0]
        assert len(idx) > 0, f"empty energy window [{lo:.3f}, {hi:.3f}] at N={N}"
        tower.append(H_evecs[idx[np.argmax(overlaps[idx])]])

    tower[N // 2] = get_zero_scar(N)[0]
    return tower


scar_tower = get_scar_tower(N, H_evals, H_evecs, psi0)

# rho0 is the ground state, which is the bottom scar, so this has to start at 1.
# fails in the first seconds if the tower picked up the wrong states
assert scar_population([rho0], scar_tower)[0] > 0.999, "ground state is not in the scar tower"

# value of the scar population in the maximally mixed state, (N+1)/D.
# pure dephasing ends here, so this is the floor, not 0
scar_pop_floor = len(scar_tower) / len(basisList)


def cell_path(bath, gamma, A):
    return os.path.join(OUTDIR, f"{bath}_N{N}", f"g{gamma:.6g}_A{A:.6g}.npz")


def bath_rates(bath, gamma):
    """(gamma_phi, gamma_1) for the chosen bath."""
    if bath == "dephasing":
        return gamma, 0.0
    if bath == "relaxation":
        return 0.0, gamma
    raise ValueError(f"unknown bath {bath!r}, expected one of {BATHS}")


def run_one(job):
    bath, gamma, A = job
    gamma_phi, gamma_1 = bath_rates(bath, gamma)
    args = {"A": A, "omega": wd}
    qargs = {"A": A, "omega": wm}

    t0 = time.perf_counter()

    # ---- scar chain ----
    c_ops = get_scar_c_ops(N, basisList, gamma_phi=gamma_phi, gamma_1=gamma_1)
    tlist, states = evolve_open(H0, H1, c_ops, rho0, args, t_max, t_idle=t_idle,
                                n_charge=n_charge, n_idle=n_idle, options=solver_options)

    # only W and ergotropy, so skip the extra eigensolves battery_series does for S and purity
    energy = np.real(qt.expect(H0, states))
    W_scar = (energy - energy[0]) / bandwidth
    erg_scar = np.array([calculate_ergotropy(r, H0, H_evals) for r in states]) / bandwidth
    scar_pop = scar_population(states, scar_tower)    # total weight in the scar tower
    del states

    # ---- decoupled qubits ----
    qc_ops = get_qubit_c_ops(gamma_phi=gamma_phi, gamma_1=gamma_1)
    _, rho_list = evolve_qubits_open(qH0_list, qH1_list, qc_ops, qargs, t_max, t_idle=t_idle,
                                     n_charge=n_charge, n_idle=n_idle, options=solver_options)
    q = qubit_battery_series(rho_list, qH0_list)

    elapsed = time.perf_counter() - t0

    # write to a temp name and rename, so a job killed mid-write never leaves
    # a half-written file that a resubmit would then skip
    path = cell_path(bath, gamma, A)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp,
             tlist=tlist, bath=bath, gamma=gamma, A=A, N=N, wd=wd, wm=wm,
             t_max=t_max, t_idle=t_idle,
             W_scar=W_scar, erg_scar=erg_scar,
             scar_pop=scar_pop, scar_pop_floor=scar_pop_floor,
             W_qubit=q["W"], erg_qubit=q["ergotropy"], erg_qubit_local=q["ergotropy_local"],
             elapsed=elapsed)
    os.replace(tmp, path)

    return bath, gamma, A, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bath", choices=BATHS, default="dephasing",
                        help="which rate goes on the horizontal axis of the grid")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute cells that already have an npz")
    cli = parser.parse_args()
    bath, overwrite = cli.bath, cli.overwrite

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

    # large A is the slowest solve, so sort those first: the slice then spreads
    # them over the array tasks, and each pool starts its longest cells first
    cells = sorted(((bath, g, A) for g in gamma_list for A in A_list),
                   key=lambda c: (-c[2], -c[1]))

    # slice BEFORE skipping finished cells, so every task agrees on who owns what
    mine = cells[task::ntask]
    jobs = [c for c in mine if overwrite or not os.path.exists(cell_path(*c))]

    os.makedirs(os.path.dirname(cell_path(bath, 0.0, 0.0)), exist_ok=True)
    print(f"N={N}  bath={bath}  task {task} of {ntask}  ncpu={ncpu}  "
          f"{len(mine)} cells owned, {len(mine) - len(jobs)} already done, {len(jobs)} to run",
          flush=True)

    if not jobs:
        raise SystemExit(0)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(ncpu, len(jobs))) as pool:
        futures = [pool.submit(run_one, j) for j in jobs]
        for f in as_completed(futures):
            b, g, A, el = f.result()
            print(f"  done  gamma={g:<8g} A={A:<6g} {el:8.1f} s", flush=True)

    print(f"{len(jobs)} cells in {time.perf_counter() - t0:.1f} s", flush=True)
