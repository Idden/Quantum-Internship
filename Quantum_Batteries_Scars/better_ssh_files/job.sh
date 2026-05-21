#!/bin/bash
#SBATCH --job-name=qbatts_test
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-0

module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

cd ~/ece_mondrag2_chi_link/itsai/qbatts

python scripts/main.py \
  --N 4 \
  --outdir /home/itsai/ece_mondrag2_chi_link/itsai/qbatts/data \
  --objective-reals 1 \
  --final-reals 1 \
  --maxiter 1 \
  --popsize 2