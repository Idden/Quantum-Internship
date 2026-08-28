#!/bin/bash
#SBATCH --job-name=qbatts
#SBATCH --account=ece_mondrag2                 # REQUIRED on ICC. check: /sw/cc.users/tools/my.accounts
#SBATCH --partition=ece_mondrag2               # REQUIRED on ICC. check: sinfo -s -o "%.25R %.12l %.12L %.5D"
#                                              # (use "secondary" only if you fit in its 4 h cap)
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.err
#SBATCH --time=04:00:00                        # set from the calibration run, +50% headroom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20                     # = popsize * 5 so one generation = one parallel wave
#SBATCH --mem-per-cpu=2G                       # scales with workers; --mem=16G silently starves 20 procs
#SBATCH --array=0-7%4                          # 8 independent DE islands, at most 4 running at once
##SBATCH --mail-type=END,FAIL                 # uncomment + set your address if you want mail
##SBATCH --mail-user=your_netid@uic.edu

set -euo pipefail

ROOT=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts
mkdir -p "$ROOT/logs"          # note: SLURM will NOT create this. if it is missing the job dies with no output.

module purge
module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

# scipy DE uses one process per worker, so every BLAS thread on top of that is
# oversubscription. main.py sets these too, but set them here so `module load`
# and any import ordering can't undo it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-none} start=$(date -Is)"
echo "cpus=${SLURM_CPUS_PER_TASK} mem/cpu=${SLURM_MEM_PER_CPU:-?}"
git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true    # provenance: which code produced this npz

python -u scripts/main.py \
  --N 12 \
  --outdir "$ROOT/data" \
  --objective-reals 4 \
  --final-reals 32 \
  --maxiter 60 \
  --popsize 4 \
  --tol 0.01

echo "end=$(date -Is)"

# ---------------------------------------------------------------------------
# Sizing this job
#
#   population P      = popsize * 5            (5 = number of DE parameters)
#   objective calls   = P * (maxiter + 1)      (upper bound; --tol can stop early)
#   simulations       = objective calls * objective-reals
#   wall time         ~ (maxiter + 1) * ceil(P / cpus-per-task) * t_eval
#
# t_eval is the cost of ONE objective call at your chosen --objective-reals.
# Get it from a calibration run first (--maxiter 1 --popsize 2, 30 min, 2 cpus):
# every objective line in the .out file already prints "elapsed=...s". Take the
# median, plug it in above, then add ~50%. Ask for the time you need and no more —
# short jobs start much sooner, and on the secondary queue 4 h is the hard cap.
# ---------------------------------------------------------------------------