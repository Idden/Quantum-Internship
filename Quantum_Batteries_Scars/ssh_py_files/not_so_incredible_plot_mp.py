import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
from multiprocessing import Pool
from quantumScarFunctions import *

N = 4
wd = 0.6365091993031
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 500

args = {"A": 0.1, "omega": wd}

H0_clean, H1, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)

OUTDIR = "/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/data"


def run_one(params):
    seed, dz, dy, dx = params

    partial_dir = os.path.join(OUTDIR, "partials")
    os.makedirs(partial_dir, exist_ok=True)

    partial_path = os.path.join(
        partial_dir,
        f"Rtau_N{N}_dz{dz}_dy{dy}_dx{dx}_seed{seed}_tmax{t_max}.npz"
    )

    if os.path.exists(partial_path):
        print(f"Skipping seed {seed}, already exists: {partial_path}", flush=True)
        return partial_path

    print(f"Starting seed={seed}, dz={dz}, dy={dy}, dx={dx}", flush=True)
    np.random.seed(seed)

    H0_dis, eigenvalues_dis, eigenstates_dis = getDisorderedScarHam(
        H0_clean,
        N,
        basisList,
        ham_disorder=[dz, dy, dx],
        fixed_seed=False
    )

    bandwidth = eigenvalues_dis[-1] - eigenvalues_dis[0]

    H = qt.QobjEvo([H0_dis, [H1, coeff]], args=args)

    psi_t = qt.sesolve(
        H,
        eigenstates_dis[0],
        tlist,
        e_ops=[H0_dis]
    )

    Rtau_scar = np.array(
        np.real(psi_t.expect[0] - psi_t.expect[0][0]) / bandwidth
    )

    tmp_path = partial_path.replace(".npz", ".tmp.npz")

    np.savez(
        tmp_path,
        seed=seed,
        tlist=tlist,
        Rtau_scar=Rtau_scar,
        N=N,
        wd=wd,
        dz=dz,
        dy=dy,
        dx=dx,
        t_max=t_max
    )

    os.replace(tmp_path, partial_path)

    print(f"Finished and saved: {partial_path}", flush=True)
    return partial_path


if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    disorders = [
        [0.3, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.3]
    ]

    os.makedirs(OUTDIR, exist_ok=True)

    seeds_per_array_job = num_cpus

    start_seed = array_id * seeds_per_array_job
    end_seed = min(start_seed + seeds_per_array_job, reals)

    seeds = list(range(start_seed, end_seed))

    print(f"Array task {array_id}", flush=True)
    print(f"Running seeds {start_seed} to {end_seed - 1}", flush=True)
    print(f"Using {num_cpus} CPUs on this node", flush=True)

    tasks = []

    for seed in seeds:
        for dz, dy, dx in disorders:
            tasks.append((seed, dz, dy, dx))

    with Pool(processes=num_cpus) as pool:
        pool.map(run_one, tasks)