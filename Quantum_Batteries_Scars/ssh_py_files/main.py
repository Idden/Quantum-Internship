import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import qutip as qt
from multiprocessing import Pool
from quantumScarFunctions import *


wd = 0.6365091993031
t_max = 200
tlist = np.linspace(0, t_max, 400)
reals = 500

OUTDIR = "/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/data"

# ADJUST NUMBERS WHEN MEET WITH IAN
xlist = np.linspace(0, 0.5, 11)
ylist = np.linspace(0, 0.5, 11)
zlist = np.linspace(0, 0.5, 11)
dslist = np.linspace(0, 2.0, 21)
ddlist = np.linspace(0, 0.5, 11)
nlist = [4, 6, 8, 10, 12, 14, 16, 18]

parameter_sweep = []

for N in nlist:
    for x in xlist:
        for y in ylist:
            for z in zlist:
                for ds in dslist:
                    for dd in ddlist:
                        parameter_sweep.append((N, x, y, z, ds, dd))

print(f"Total parameter points: {len(parameter_sweep)}", flush=True)


def safe_float_name(x):
    return f"{x:.4f}".replace(".", "p")

H0_clean, H1_clean, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
def run_one(params):

    args = {"A": ds, "omega": wd}

    seed, N, x, y, z, ds, dd = params

    partial_dir = os.path.join(OUTDIR, "partials")
    os.makedirs(partial_dir, exist_ok=True)

    partial_path = os.path.join(
        partial_dir,
        f"Rtau_N{N}"
        f"_x{safe_float_name(x)}"
        f"_y{safe_float_name(y)}"
        f"_z{safe_float_name(z)}"
        f"_ds{safe_float_name(ds)}"
        f"_dd{safe_float_name(dd)}"
        f"_seed{seed}"
        f"_tmax{t_max}.npz"
    )

    if os.path.exists(partial_path):
        print(f"Skipping existing: {partial_path}", flush=True)
        return partial_path

    print(
        f"Starting seed={seed}, N={N}, x={x}, y={y}, z={z}, ds={ds}, dd={dd}",
        flush=True
    )

    np.random.seed(seed)

    H0_dis, eigenvalues_dis, eigenstates_dis = getDisorderedScarHam(
        H0_clean,
        N,
        basisList,
        ham_disorder=[z, y, x],
        fixed_seed=False
    )
    H1, disorder_weights = getDisorderedScarH1(N, basisList, ds_dis=dd)

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
        x=x,
        y=y,
        z=z,
        ds=ds,
        dd=dd,
        t_max=t_max
    )

    os.replace(tmp_path, partial_path)

    print(f"Finished and saved: {partial_path}", flush=True)
    return partial_path


if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    os.makedirs(OUTDIR, exist_ok=True)

    total_jobs = len(parameter_sweep) * reals

    jobs_per_array_task = num_cpus

    start_job = array_id * jobs_per_array_task
    end_job = min(start_job + jobs_per_array_task, total_jobs)

    print(f"Array task {array_id}", flush=True)
    print(f"Using {num_cpus} CPUs", flush=True)
    print(f"Running global jobs {start_job} to {end_job - 1}", flush=True)
    print(f"Total jobs: {total_jobs}", flush=True)

    tasks = []

    for global_job_id in range(start_job, end_job):
        param_index = global_job_id // reals
        seed = global_job_id % reals

        N, x, y, z, ds, dd = parameter_sweep[param_index]

        tasks.append((seed, N, x, y, z, ds, dd))

    with Pool(processes=num_cpus) as pool:
        pool.map(run_one, tasks)