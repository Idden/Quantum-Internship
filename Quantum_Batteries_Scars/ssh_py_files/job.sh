#!/bin/bash
#SBATCH --job-name=qbatts
#SBATCH --output=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.out
#SBATCH --error=/home/itsai/ece_mondrag2_chi_link/itsai/qbatts/logs/qbatts_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-31%16

module load python39
source ~/ece_mondrag2_chi_link/itsai/envs/qenv/bin/activate

cd ~/ece_mondrag2_chi_link/itsai/qbatts

python scripts/main.py