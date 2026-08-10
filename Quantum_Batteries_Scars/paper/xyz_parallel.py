import os
os.environ["OMP_NUM_THREADS"] = "1"   # before numpy, so workers don't fight over cores

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from helper.quantumScarFunctions import *
import time

# print(os.process_cpu_count())

N = 14
wd = 0.6366896896896898
wm = 1.0
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 500
dis = 0.3

dz = 0.0
dy = 0.0
dx = dis

args = {"A": 0.1, "omega": wd}
qargs = {"A": 0.1, "omega": wm}

H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
H1, driveWeights = get_scar_H1(N, basisList)

def run_one(seed):
    np.random.seed(seed)

    H0, eigenvalues, eigenstates = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[dz, dy, dx])
    bandwidth = eigenvalues[-1] - eigenvalues[0]

    H = qt.QobjEvo([H0, [H1, coeff]], args=args)
    psi_t = qt.sesolve(H, eigenstates[0], tlist, e_ops=[H0])
    Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm, ham_disorder=[dz, dy, dx])

    dE_tot = 0
    bw_tot = 0
    for qH0, qH1 in zip(qH0_list, qH1_list):
        qeigenvalues, qeigenstates = qH0.eigenstates()
        bw_tot += qeigenvalues[-1] - qeigenvalues[0]

        qH = qt.QobjEvo([qH0, [qH1, coeff]], args=qargs)
        qpsi_t = qt.sesolve(qH, qeigenstates[0], tlist, e_ops=[qH0])
        dE_tot = dE_tot + np.real(qpsi_t.expect[0] - qpsi_t.expect[0][0])

    Rtau_qubit = dE_tot / bw_tot

    return Rtau_scar, Rtau_qubit

if __name__ == "__main__":
    seeds = np.random.SeedSequence(0).generate_state(reals)

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, seeds))
    elapsed = time.perf_counter() - t0

    print(f"{reals} realizations in {elapsed:.1f} s "
      f"({elapsed/reals:.5f} s per realization)")

    full_scar = np.mean([r[0] for r in results], axis=0)
    full_qubit = np.mean([r[1] for r in results], axis=0)

    np.savez(f"xyz_data/x_dis_N{N}.npz", tlist=tlist, scar=full_scar, qubit=full_qubit)

    plt.title(f"Avged Rtau for Disorder=[{dz},{dy},{dx}] for N={N}")
    plt.xlabel("Time")
    plt.ylabel("Rtau")
    plt.plot(tlist, full_scar, label="Scar")
    plt.plot(tlist, full_qubit, label="Qubit")
    plt.legend()
    plt.ylim(0, 1)
    # plt.savefig(f"figures/xyz_N{N}_dis{dis}_reals{reals}.pdf")
    plt.show()