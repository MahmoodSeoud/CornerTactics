#!/bin/bash
#SBATCH --job-name=xgb_trk
#SBATCH --output=logs/xgboost_posonly_tracking_%j.out
#SBATCH --error=logs/xgboost_posonly_tracking_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "XGBoost position-only: tracking data only (multi-seed + permutation)"
echo "Job ID: $SLURM_JOB_ID"
date

python -u scripts/statsbomb_xgboost_baseline.py --no-statsbomb --tracking-posonly --permutation

echo "Done"
date
