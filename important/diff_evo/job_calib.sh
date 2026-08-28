#!/bin/bash
# ---------------------------------------------------------------------------
# STEP 2 of 3 -- calibration. Measures how long ONE objective evaluation takes
# on this cluster, so job_de.sh can ask for the right walltime.
#
#   sbatch job_calib.sh
#
# Then read the median off the log:
#
#   grep -o 'elapsed=[0-9.]*' logs/calib_*.out | cut -d= -f2 | sort -n | \
#     awk '{a[NR]=$1} END {print "median t_eval =", a[int(NR/2)+1], "s"}'
#
# Ask for short walltime and you start sooner. Do not skip this step and
# guess: evaluation cost varies by ~20x across the search space, because the
# integrator has to resolve the drive, and the cost rises steeply with the
# drive amplitude ds (measured correlation 0.89).
# ---------------------------------------------------------------------------
#SBATCH --job-name=qb_calib
#SBATCH --account=ece_mondrag2
#SBATCH --partition=ece_mondrag2
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/calib_%j.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/calib_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

set -euo pipefail

ROOT=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts
mkdir -p "$ROOT/logs"

module purge
module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

# scipy's DE uses one PROCESS per worker, so any BLAS thread on top of that is
# oversubscription. main.py sets these too, but set them here so module load
# and import ordering cannot undo it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ROOT/scripts"

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"

python -u main.py \
  --N 12 \
  --cache "$ROOT/cache" \
  --outdir "$ROOT/calib" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --objective-reals 8 \
  --final-reals 8 \
  --maxiter 3 \
  --popsize 2 \
  --nt 1601

echo "end=$(date -Is)"
echo "--- t_eval distribution ---"
grep -o 'elapsed=[0-9.]*' "$ROOT/logs/calib_${SLURM_JOB_ID}.out" | cut -d= -f2 \
  | sort -n | awk '{a[NR]=$1} END {
      print "n       =", NR;
      print "median  =", a[int(NR/2)+1], "s";
      print "p90     =", a[int(NR*0.9)+1], "s";
      print "max     =", a[NR], "s" }'
