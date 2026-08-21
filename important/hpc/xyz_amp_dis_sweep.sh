#!/bin/bash
#SBATCH --job-name=xyz_amp_dis_sweep
#SBATCH --account=ece_mondrag2_chi
#SBATCH --partition=batch
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

# module load python
# source ~/venvs/scar/bin/activate

# one thread per worker, or BLAS oversubscribes the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p xyz_amp_data/parts

python xyz_amp_dis_sweep.py