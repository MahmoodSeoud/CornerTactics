#!/bin/bash
#SBATCH --job-name=xgb_posonly
#SBATCH --output=logs/xgboost_posonly_%j.out
#SBATCH --error=logs/xgboost_posonly_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge
#SBATCH --nodelist=cn7

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "XGBoost position-only: StatsBomb + tracking (multi-seed + permutation)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

python -u scripts/statsbomb_xgboost_baseline.py --all --permutation

echo "Done"
date
