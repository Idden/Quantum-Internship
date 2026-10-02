#!/bin/bash
#SBATCH --job-name=deco_drive_sweep
#SBATCH --account=ece_mondrag2_chi
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

# 36 cells (6 gamma x 6 A) / 3 array tasks = 12 cells per task = 1 per cpu.
# if you change gamma_list / A_list, keep (array size) x (cpus-per-task) >= number of cells.
#
# memory: evolve_open stores every rho(t), ~ n_charge * D^2 * 16 bytes per worker
#   N=10 (D=123): ~0.1 GB    N=12 (D=322): ~0.8 GB    N=14 (D=843): ~5.7 GB
#
# submit from this folder (logs/ must exist BEFORE sbatch, or slurm has nowhere to write):
#   mkdir -p logs
#   sbatch --export=ALL,BATH=dephasing  decoherence_drive_sweep.sh
#   sbatch --export=ALL,BATH=relaxation decoherence_drive_sweep.sh
# resubmitting the same command after a timeout only runs the cells that are missing.

# module load python
# source ~/venvs/scar/bin/activate

# one thread per worker, or BLAS oversubscribes the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python decoherence_drive_sweep.py --bath "${BATH:-dephasing}"
