import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from helper.quantumScarFunctions import *
import time

N = 8
wd = 0.6366896896896898
reals = 1000
dlist = np.linspace(0, 5.0, 201)

dz = 0
dy = 0
dx = 0

H0_clean, eigenvalues_clean, eigenstates_clean, psi0, basisList = get_scar_ham(N)

def run_one(seed):
    np.random.seed(seed)
    out = []

    for dz in dlist:
        H0_dis, eigenvalues, eigenstates = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[dz, dy, dx])

        temp = 0
        for i in range(len(eigenvalues) - 2):
            deltaE0 = eigenvalues[i+1] - eigenvalues[i]
            deltaE1 = eigenvalues[i+2] - eigenvalues[i+1]
            temp += np.min([deltaE0, deltaE1]) / np.max([deltaE0, deltaE1])

        out.append(temp / (len(eigenvalues) - 2))

    return out

if __name__ == "__main__":
    seeds = np.random.SeedSequence(0).generate_state(reals)

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, seeds))
    elapsed = time.perf_counter() - t0

    print(f"{reals} realizations in {elapsed:.1f} s "
      f"({elapsed/reals:.5f} s per realization)")

    rn_all = np.array(results)
    rn = rn_all.mean(axis=0)

    np.savez(f"rn_data/rn_z_N{N}.npz", dlist=dlist, rn=rn, rn_all=rn_all)