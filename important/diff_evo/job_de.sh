#!/bin/bash
# ---------------------------------------------------------------------------
# STEP 3 of 3 -- the differential evolution search.
#
#   sbatch job_de.sh
#
# 8 independent DE islands (the array index seeds each one), at most 4 running
# at once. Islands do not communicate; that is the point -- DE is a stochastic
# search and one run's optimum is not evidence. Compare the islands, then use
# rerank.py to pool every evaluation from all of them.
#
# BEFORE SUBMITTING
#   1. job_cache.sh has finished and $ROOT/cache/ contains struct_N12.npz
#   2. job_calib.sh has finished and you have read the median t_eval off it
#   3. --time below has been set from that median (see the sizing block at the
#      bottom of this file), not left at whatever it says now
# ---------------------------------------------------------------------------
#SBATCH --job-name=qb_de
#SBATCH --account=ece_mondrag2
#SBATCH --partition=ece_mondrag2
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/de_%A_%a.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/de_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20        # = popsize * n_params, so one generation is one wave
#SBATCH --mem-per-cpu=2G          # per CPU, not per job: the DE workers are processes
#SBATCH --array=0-7%4
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=itsai@uic.edu

set -euo pipefail

ROOT=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts
mkdir -p "$ROOT/logs"            # SLURM will NOT create this; if it is missing
                                 # the job dies with nowhere to write

module purge
module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ROOT/scripts"

echo "host=$(hostname) job=${SLURM_JOB_ID} island=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
echo "cpus=${SLURM_CPUS_PER_TASK} mem/cpu=${SLURM_MEM_PER_CPU:-?}"
git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true   # which code made this npz

python -u main.py \
  --N 12 \
  --cache "$ROOT/cache" \
  --outdir "$ROOT/data" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --objective-reals 8 \
  --final-reals 32 \
  --maxiter 120 \
  --popsize 4 \
  --tol 0.005 \
  --nt 1601

echo "end=$(date -Is)"

# ---------------------------------------------------------------------------
# SIZING
#
#   n_params        = 5, or 6 with --search-wd, or 7 with --search-wd --search-wq
#   population P    = max(5, popsize * n_params)      <- scipy's rule
#   objective calls = P * (maxiter + 1)               <- upper bound, --tol may stop early
#   wall time       ~ (maxiter + 1) * ceil(P / cpus-per-task) * t_eval
#
#   Set --cpus-per-task = P so one generation is exactly one parallel wave.
#     5 params, popsize 4 -> P = 20 -> --cpus-per-task=20
#     7 params, popsize 4 -> P = 28 -> --cpus-per-task=28
#
#   t_eval is the cost of ONE objective call at your --objective-reals, from
#   job_calib.sh. It scales roughly linearly in --objective-reals and rises
#   steeply with the drive amplitude ds, so use the p90 rather than the median
#   if you want the job to survive an unlucky population. Add 50% on top.
#
#   Worked example, 5 params, t_eval = 6 s at 8 realizations:
#     121 waves * 6 s = 726 s ~ 12 min, +50% -> ask for 00:30:00.
#   Short jobs start much sooner. On the `secondary` queue 4 h is a hard cap.
#
# WHY --objective-reals 8 AND NOT 1
#   At 1 realization the score is dominated by disorder noise: across 8 seeds
#   at fixed parameters the spread exceeded the effect, and only 4-6 seeds out
#   of 8 even had the right sign. DE with one realization returns whichever
#   point won the disorder lottery. The rebuild overhead that used to make more
#   realizations unaffordable is gone, so spend the budget here rather than on
#   more generations.
#
# TO SEARCH THE DRIVE FREQUENCIES
#   add   --search-wd            (6 params, set --cpus-per-task=24)
#   or    --search-wd --search-wq  (7 params, set --cpus-per-task=28)
#   Disorder shifts the effective scar gap, so the clean-chain wd = 0.6367 is
#   off resonance at large x/y/z; --search-wq stops the comparison qubits from
#   being handicapped the same way.
# ---------------------------------------------------------------------------
