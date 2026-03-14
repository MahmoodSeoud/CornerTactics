#!/bin/bash
#SBATCH --job-name=perm_posonly
#SBATCH --output=logs/perm_position_only_%j.out
#SBATCH --error=logs/perm_position_only_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "Starting permutation test: USSF position-only ablation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

python scripts/permutation_test_position_only.py

echo "Done"
date
