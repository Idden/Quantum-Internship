#!/bin/bash
#SBATCH --job-name=scar-bands
#SBATCH --account=ece_mondrag2
#SBATCH --partition=ece_mondrag2
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Submit this only AFTER make_scar_states.sh has finished - it reads the npz
# that job writes. N must match make_scar_states.py.
#
# Run "mkdir -p logs" ONCE before your first sbatch. Slurm opens the --output
# file before this script starts, so it dies with nowhere to write otherwise.
#
#   sbatch xyz_parallel.sh

# module load python
# source ~/venvs/scar/bin/activate

# one thread per worker, or BLAS oversubscribes the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p xyz_data/parts

python xyz_parallel.py