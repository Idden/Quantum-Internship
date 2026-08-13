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
dis = 1.0

args = {"A": 0.1, "omega": wd}
qargs = {"A": 0.1, "omega": wm}

H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
bandwidth = eigenvalues[-1] - eigenvalues[0]

def run_one(seed):
    np.random.seed(seed)

    H1, _ = get_scar_H1(N, basisList, ds_dis=dis)

    H = qt.QobjEvo([H0_clean, [H1, coeff]], args=args)
    psi_t = qt.sesolve(H, eigenstates[0], tlist, e_ops=[H0_clean])
    Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm, ds_dis=dis)

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

    full_scar = np.array([r[0] for r in results])
    full_qubit = np.array([r[1] for r in results])

    plt.title(f"Drive Strength Disorder")
    plt.xlabel("Time")
    plt.ylabel(r"$R(\tau)$")
    for arr, lab in [(full_scar, "Scar"), (full_qubit, "Qubit")]:
        m, sem = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        print(max(sem))
        line, = plt.plot(tlist, m, label=lab)
        plt.fill_between(tlist, m - sem, m + sem,
                         color=line.get_color(), alpha=0.3, lw=0)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"figures/ds_dis_N{N}_dis{dis}_reals{reals}.pdf")
    plt.show()