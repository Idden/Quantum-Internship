#!/bin/bash
#SBATCH --job-name=scar-states
#SBATCH --account=ece_mondrag2
#SBATCH --partition=ece_mondrag2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# N is set at the top of make_scar_states.py. Keep it the same in
# xyz_parallel.py and merge_bands.py.
#
# Run "mkdir -p logs" ONCE before your first sbatch. Slurm opens the --output
# file before this script starts, so it dies with nowhere to write otherwise.
#
#   sbatch make_scar_states.sh

# module load python
# source ~/venvs/scar/bin/activate

# this job WANTS threads: one dense eigh and one sparse LU
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

mkdir -p xyz_data

python make_scar_states.py