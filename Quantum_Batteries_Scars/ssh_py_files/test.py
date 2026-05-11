import os
import numpy as np
import qutip as qt
from multiprocessing import Pool
from quantumScarFunctions import *

N = 4
wd = 0.6365091993031
wm = 1.0
freq_dis = 0.00
indv_qubit = False
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 8
rand = True
z_ham = False
dis = 0.3

args = {"A":0.1, "omega":wd}
qargs = {"A":0.1, "omega":wm}

H0_clean, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
qH0_clean, qH1, qeigenvalues, qeigenstates = get_qubit_ham(N)

dx = dis
dy = 0.0
dz = 0.0

def run_one(seed):
    print(f"Starting seed {seed}", flush=True)
    np.random.seed(seed)

    # -----------------------
    # Scar realization
    # -----------------------
    H0_dis, eigenvalues_dis, eigenstates_dis = getDisorderedScarHam(
        H0_clean,
        N,
        basisList,
        ham_disorder=[dz, dy, dx],
        fixed_seed=False
    )

    bandwidth = eigenvalues_dis[-1] - eigenvalues_dis[0]

    H = qt.QobjEvo([H0_dis, [H1, coeff]], args=args)
    psi_t = qt.sesolve(H, eigenstates_dis[0], tlist, e_ops=[H0_dis])

    Rtau_scar = np.array(
        np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth
    )

    # -----------------------
    # Qubit realization
    # -----------------------
    qH0_dis, qeigenvalues_dis, qeigenstates_dis = getDisorderedQubitHam(
        qH0_clean,
        N,
        ham_disorder=[dz, dy, dx],
        fixed_seed=False
    )

    qbandwidth = qeigenvalues_dis[-1] - qeigenvalues_dis[0]

    qH = qt.QobjEvo([qH0_dis, [qH1, coeff]], args=qargs)
    qpsi_t = qt.sesolve(qH, qeigenstates_dis[0], tlist, e_ops=[qH0_dis])

    Rtau_qubit = np.array(
        np.real(qpsi_t.expect[0] - qpsi_t.expect[0][0]) / qbandwidth
    )

    print(f"Finished seed {seed}", flush=True)

    return Rtau_scar, Rtau_qubit

if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_cpus} CPU processes", flush=True)

    seeds = list(range(reals))

    with Pool(processes=num_cpus) as pool:
        results = pool.map(run_one, seeds)

    print("Pool worked", flush=True)