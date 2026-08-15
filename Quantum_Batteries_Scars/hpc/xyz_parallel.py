import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
from concurrent.futures import ProcessPoolExecutor
from quantumScarFunctions import *
import time

assert int(qt.__version__.split(".")[0]) >= 5, (
    "needs qutip 5: the coefficient functions here use the f(t, A, omega) "
    "keyword signature, qutip 4 only accepts f(t, args)"
)

N = int(os.environ.get("SCAR_N", 20))
wd = 0.6366896896896898
wm = 1.0
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 100
dis = 0.3

# all three axes in one run. label -> [dz, dy, dx]
configs = {
    "z": [dis, 0.0, 0.0],
    "y": [0.0, dis, 0.0],
    "x": [0.0, 0.0, dis],
}

args = {"A": 0.1, "omega": wd}
qargs = {"A": 0.1, "omega": wm}

H0_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N, diagonalize=False)
H1, driveWeights = get_scar_H1(N, basisList)

_scarFile = f"xyz_data/scar_states_N{N}.npz"
assert os.path.exists(_scarFile), f"run make_scar_states.py with SCAR_N={N} first"
_sc = np.load(_scarFile)

scarMatC = np.ascontiguousarray(
    _sc["scarMatC"] if "scarMatC" in _sc.files else _sc["scarMat"].conj()
)
assert scarMatC.shape[1] == len(basisList), (
    f"scar file has D={scarMatC.shape[1]} but N={N} gives D={len(basisList)}"
)


def gs_and_bw(H):
    # only the ground state and the bandwidth are ever used, so don't build the
    # whole spectrum. this is the only reason N=20+ fits in memory.
    e_low, v_low = H.eigenstates(sparse=True, sort="low", eigvals=1)
    e_high = H.eigenenergies(sparse=True, sort="high", eigvals=1)

    gap = e_high[0] - e_low[0]
    assert abs(np.imag(gap)) < 1e-10, f"non-real bandwidth: {gap}"

    return v_low[0], float(np.real(gap))


def run_one(job):
    label, seed = job
    dz, dy, dx = configs[label]

    np.random.seed(seed)

    H0, _, _ = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[dz, dy, dx], diagonalize=False)
    psi_i, bandwidth = gs_and_bw(H0)

    H = qt.QobjEvo([H0, [H1, coeff]], args=args)

    def scar_prob(t, psi):
        return float(np.sum(np.abs(scarMatC @ psi.full().ravel())**2))

    psi_t = qt.sesolve(H, psi_i, tlist, e_ops=[H0, scar_prob])
    scarProb = np.array(np.real(psi_t.expect[1]))
    Rtau_scar = np.array(np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth)

    np.random.seed(seed)   # same disorder for the qubits as for the scar chain
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

    return label, seed, Rtau_scar, Rtau_qubit, scarProb


if __name__ == "__main__":
    seeds = np.random.SeedSequence(0).generate_state(reals)

    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    # SLURM_ARRAY_TASK_COUNT is not set on every slurm build. if it silently
    # falls back to 1, all 20 tasks compute all 300 realizations and
    # merge_bands trips its duplicate-seed assert after the whole array ran.
    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        ntask = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    elif "SLURM_ARRAY_TASK_MAX" in os.environ:
        ntask = (int(os.environ["SLURM_ARRAY_TASK_MAX"])
                 - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0)) + 1)
    else:
        ntask = 1
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # every (axis, seed) pair, sliced so task k takes every ntask-th one
    jobs = [(label, int(s)) for label in configs for s in seeds][task::ntask]
    print(f"N={N}  task {task}/{ntask}  ncpu={ncpu}  {len(jobs)} realizations", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=ncpu) as pool:
        results = list(pool.map(run_one, jobs))
    elapsed = time.perf_counter() - t0

    print(f"{len(jobs)} realizations in {elapsed:.1f} s "
          f"({elapsed/len(jobs):.5f} s per realization)")

    os.makedirs("xyz_data/parts", exist_ok=True)

    for label in configs:
        keep = [r for r in results if r[0] == label]
        if not keep:
            continue
        np.savez(f"xyz_data/parts/{label}_dis_N{N}_task{task:04d}.npz",
                 tlist=tlist,
                 seeds=np.array([r[1] for r in keep]),
                 scar=np.array([r[2] for r in keep]),
                 qubit=np.array([r[3] for r in keep]),
                 scarprob=np.array([r[4] for r in keep]))