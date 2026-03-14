#!/bin/bash
#SBATCH --job-name=sb_ussf
#SBATCH --output=logs/statsbomb_ussf_%j.out
#SBATCH --error=logs/statsbomb_ussf_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge
#SBATCH --nodelist=cn7

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "Starting StatsBomb USSF pipeline (eval + permutation test)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

python -u scripts/statsbomb_ussf_pipeline.py --permutation

echo "Done"
date
