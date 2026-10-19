import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
from concurrent.futures import ProcessPoolExecutor
from helper.quantumScarFunctions import *
from helper.quantumScarsPlotting import *
import time

N = 14
tau_r = get_tau_r(N)
wd = 2*np.pi / tau_r
wm = 1.0
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 500
dis = 0.3

dz = 0.0
dy = 0.0
dx = dis

if dz != 0.0:
    dis_type = "z"
elif dy != 0.0:
    dis_type = "y"
elif dx != 0.0:
    dis_type = "x"
else:
    dis_type = "clean"

args = {"A": 0.1, "omega": wd}
qargs = {"A": 0.1, "omega": wm}

H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
H1, driveWeights = get_scar_H1(N, basisList)
scarIndices, scarStates = get_scar_states(N, psi0)

def run_one(seed):
    H0, _, _ = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[dz, dy, dx], fixed_seed=seed, diagonalize=False)
    e_low, v_low = H0.eigenstates(sparse=True, sort="low", eigvals=1)
    e_high = H0.eigenenergies(sparse=True, sort="high", eigvals=1)
    bandwidth = e_high[0] - e_low[0]
    gs = v_low[0]

    H = qt.QobjEvo([H0, [H1, coeff]], args=args)
    psi_t = qt.sesolve(H, gs, tlist, e_ops=[H0], options={"store_states": True})
    scarProbs = scar_overlap_from_states(N, tlist, scarStates, psi_t)
    Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm, ham_disorder=[dz, dy, dx], fixed_seed=seed)

    dE_tot = 0
    bw_tot = 0
    for qH0, qH1 in zip(qH0_list, qH1_list):
        qeigenvalues, qeigenstates = qH0.eigenstates()
        bw_tot += qeigenvalues[-1] - qeigenvalues[0]

        qH = qt.QobjEvo([qH0, [qH1, coeff]], args=qargs)
        qpsi_t = qt.sesolve(qH, qeigenstates[0], tlist, e_ops=[qH0])
        dE_tot = dE_tot + np.real(qpsi_t.expect[0] - qpsi_t.expect[0][0])

    Rtau_qubit = dE_tot / bw_tot

    return Rtau_scar, Rtau_qubit, scarProbs

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xyz_data")
    os.makedirs(out_dir, exist_ok=True)

    seeds = np.random.SeedSequence(0).generate_state(reals)

    n_workers = os.cpu_count()
    print(f"Running with {n_workers} worker processes")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(run_one, seeds))
    elapsed = time.perf_counter() - t0

    print(f"{reals} realizations in {elapsed:.1f} s "
      f"({elapsed/reals:.5f} s per realization)")

    full_scar = np.array([r[0] for r in results])
    full_qubit = np.array([r[1] for r in results])
    full_scarProbs = np.array([r[2] for r in results])

    np.savez(f"xyz_data/{dis_type}_dis_N{N}_error_bands.npz", tlist=tlist, scar=full_scar, qubit=full_qubit, scarProbs=full_scarProbs)