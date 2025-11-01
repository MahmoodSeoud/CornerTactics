#!/bin/bash
#SBATCH --job-name=config1_v100
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/config1_v100_%j.out
#SBATCH --error=logs/config1_v100_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

echo "=========================================="
echo "EXPERIMENT CONFIG 1: Reduced Shot Weight (0.3)"
echo "GPU: V100"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

pip install xgboost --quiet

python scripts/training/train_baseline_config1.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="