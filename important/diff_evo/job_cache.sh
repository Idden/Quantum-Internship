#!/bin/bash
# ---------------------------------------------------------------------------
# STEP 1 of 3 -- build the preprocessing caches. Run this ONCE per N, before
# anything else. It is short; submit it and wait for it to finish.
#
#   sbatch job_cache.sh
#
# Writes $ROOT/cache/struct_N{N}.npz and $ROOT/cache/seeds_N{N}.npz.
# The DE array tasks load these instead of rebuilding the basis, the flip
# pattern, the clean Hamiltonian and the E=0 scar in every worker process of
# every task.
# ---------------------------------------------------------------------------
#SBATCH --job-name=qb_cache
#SBATCH --account=ece_mondrag2
#SBATCH --partition=ece_mondrag2
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/cache_%j.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/cache_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

ROOT=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts
mkdir -p "$ROOT/logs" "$ROOT/cache"

module purge
module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

export PYTHONUNBUFFERED=1
cd "$ROOT/scripts"

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true

# The equivalence check against quantumScarFunctions.py runs here, against the
# REAL qutip in this venv. If it fails, stop and fix it -- do not submit the
# search. It is the guard against quantumScarFunctions.py drifting under the
# fast path, which has silently cost this project a whole allocation before.
python -u validate_core.py

# Build more seeds than the search uses. They cost nothing (4 x N floats each)
# and the re-ranking pass at 32-64 realizations is what turns a noisy DE
# optimum into a number worth quoting.
python -u build_cache.py --N 12 --max-seeds 128 --outdir "$ROOT/cache"

echo "end=$(date -Is)"
ls -la "$ROOT/cache"
