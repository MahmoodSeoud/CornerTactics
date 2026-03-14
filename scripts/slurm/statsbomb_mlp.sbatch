#!/bin/bash
#SBATCH --job-name=sb_mlp
#SBATCH --output=logs/statsbomb_mlp_%j.out
#SBATCH --error=logs/statsbomb_mlp_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "StatsBomb MLP baseline (multi-seed + permutation)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

python -u scripts/statsbomb_mlp_baseline.py --permutation

echo "Done"
date
