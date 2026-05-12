import os
import numpy as np

N = 4
wd = 0.6365091993031
t_max = 200
reals = 500

OUTDIR = "/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/data"
PARTIAL_DIR = os.path.join(OUTDIR, "partials")

disorders = [
    [0.3, 0.0, 0.0],
    [0.0, 0.3, 0.0],
    [0.0, 0.0, 0.3]
]

for dz, dy, dx in disorders:
    scar_results = []
    seeds_loaded = []
    tlist = None

    for seed in range(reals):
        path = os.path.join(
            PARTIAL_DIR,
            f"Rtau_N{N}_dz{dz}_dy{dy}_dx{dx}_seed{seed}_tmax{t_max}.npz"
        )

        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        data = np.load(path)
        scar_results.append(data["Rtau_scar"])
        seeds_loaded.append(seed)

        if tlist is None:
            tlist = data["tlist"]

    scar_results = np.array(scar_results)
    avged_scar = np.mean(scar_results, axis=0)

    save_path = os.path.join(
        OUTDIR,
        f"Rtau_N{N}_dz{dz}_dy{dy}_dx{dx}_reals{len(seeds_loaded)}_tmax{t_max}.npz"
    )

    np.savez(
        save_path,
        tlist=tlist,
        scar_results=scar_results,
        avged_scar=avged_scar,
        seeds_loaded=np.array(seeds_loaded),
        N=N,
        wd=wd,
        dz=dz,
        dy=dy,
        dx=dx,
        reals=len(seeds_loaded),
        t_max=t_max
    )

    print(f"Saved averaged file: {save_path}")
    print(f"Loaded {len(seeds_loaded)} / {reals} realizations")