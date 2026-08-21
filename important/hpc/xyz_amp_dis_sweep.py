import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
from concurrent.futures import ProcessPoolExecutor
from quantumScarFunctions import *
import time

N = 4
wd = 0.6366896896896898
wm = 1.0
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 200
xyzdis_list = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
ampdis_list = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]

args = {"A": 0.1, "omega": wd}
qargs = {"A": 0.1, "omega": wm}

H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N, diagonalize=False)

# unit direction per axis, scaled by xyzdis inside run_one
configs = {
    "z": [1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "x": [0.0, 0.0, 1.0],
}


def gs_and_bw(H):
    # only the ground state and the bandwidth are ever used, so don't build the
    # whole spectrum. this is the only reason N=20+ fits in memory.
    e_low, v_low = H.eigenstates(sparse=True, sort="low", eigvals=1)
    e_high = H.eigenenergies(sparse=True, sort="high", eigvals=1)

    # bandwidth is the difference between the highest and lowest eigenvalues
    gap = e_high[0] - e_low[0]
    assert abs(np.imag(gap)) < 1e-10, f"non-real bandwidth: {gap}"

    return v_low[0], float(np.real(gap))


def run_one(job):
    label, xyzdis, ampdis, seed = job
    uz, uy, ux = configs[label]
    dz, dy, dx = uz * xyzdis, uy * xyzdis, ux * xyzdis

    np.random.seed(seed)

    H0, _, _ = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[dz, dy, dx], diagonalize=False)
    H1, _ = get_scar_H1(N, basisList, ds_dis=ampdis)
    psi_i, bandwidth = gs_and_bw(H0)

    H = qt.QobjEvo([H0, [H1, coeff]], args=args)

    psi_t = qt.sesolve(H, psi_i, tlist, e_ops=[H0])
    Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    np.random.seed(seed)   # same disorder for the qubits as for the scar chain
    qH0_list, qH1_list, _ = get_qubit_ham(N, wm=wm, ham_disorder=[dz, dy, dx], ds_dis=ampdis)

    dE_tot = 0
    bw_tot = 0
    for qH0, qH1 in zip(qH0_list, qH1_list):
        qeigenvalues, qeigenstates = qH0.eigenstates()
        bw_tot += qeigenvalues[-1] - qeigenvalues[0]

        qH = qt.QobjEvo([qH0, [qH1, coeff]], args=qargs)
        qpsi_t = qt.sesolve(qH, qeigenstates[0], tlist, e_ops=[qH0])
        dE_tot = dE_tot + np.real(qpsi_t.expect[0] - qpsi_t.expect[0][0])

    Rtau_qubit = dE_tot / bw_tot

    return label, xyzdis, ampdis, seed, Rtau_scar, Rtau_qubit


if __name__ == "__main__":
    seeds = np.random.SeedSequence(0).generate_state(reals)

    # off the cluster these all fall back to "task 0 of 1", i.e. run everything
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # SLURM_ARRAY_TASK_COUNT is not set on every slurm build. if it silently
    # fell back to 1, all 20 array tasks would compute all 300 realizations and
    # merge_bands would only catch it, on its duplicate-seed assert, after the
    # whole array had already run.
    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        ntask = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    elif "SLURM_ARRAY_TASK_MAX" in os.environ:
        ntask = (int(os.environ["SLURM_ARRAY_TASK_MAX"])
                 - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0)) + 1)
    else:
        ntask = 1

    # number of cores used by in an array task. if not set, fall back to the number of cores on the machine.
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # every (axis, xyzdis, ampdis, seed) pair, sliced so task k takes every ntask-th one
    jobs = [(label, xyzdis, ampdis, int(s))
            for label in configs
            for xyzdis in xyzdis_list
            for ampdis in ampdis_list
            for s in seeds][task::ntask]
    print(f"N={N}  task {task} of {ntask}  ncpu={ncpu}  {len(jobs)} realizations", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=ncpu) as pool:
        results = list(pool.map(run_one, jobs))
    elapsed = time.perf_counter() - t0

    print(f"{len(jobs)} realizations in {elapsed:.1f} s "
          f"({elapsed/len(jobs):.5f} s per realization)")

    os.makedirs("xyz_amp_data/parts", exist_ok=True)

    for label in configs:
        keep = [r for r in results if r[0] == label]
        if not keep:
            continue
        np.savez(f"xyz_amp_data/parts/{label}_dis_N{N}_task{task:04d}.npz",
                 tlist=tlist,
                 xyzdis=np.array([r[1] for r in keep]),
                 ampdis=np.array([r[2] for r in keep]),
                 seeds=np.array([r[3] for r in keep]),
                 scar=np.array([r[4] for r in keep]),
                 qubit=np.array([r[5] for r in keep])
        )