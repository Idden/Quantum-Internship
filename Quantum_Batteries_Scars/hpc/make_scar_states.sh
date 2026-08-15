#!/bin/bash
#SBATCH --job-name=scar-bands
#SBATCH --account=CHANGEME
#SBATCH --partition=CHANGEME
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

# IMPORTANT: run this ONCE from the login node before the first sbatch.
# slurm opens the --output file before any line of this script executes,
# so the mkdir below cannot save you:
#   mkdir -p logs xyz_data/parts

# module load python
# source ~/venvs/scar/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# single source of truth for N, read by all three python scripts
export SCAR_N=20

mkdir -p logs xyz_data/parts

python xyz_parallel.py

# then, once the whole array is done:
#   SCAR_N=20 python merge_bands.py