#!/bin/bash
#SBATCH --job-name=scar-bands
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

# module load python
# source ~/venvs/scar/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs xyz_data/parts

python xyz_parallel.py

# then, once the whole array is done:
#   python merge_bands.py