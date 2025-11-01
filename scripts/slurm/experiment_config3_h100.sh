#!/bin/bash
#SBATCH --job-name=config3_h100
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/config3_h100_%j.out
#SBATCH --error=logs/config3_h100_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

echo "=========================================="
echo "EXPERIMENT CONFIG 3: Combined Approach"
echo "GPU: H100"
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

python scripts/training/train_baseline_config3.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="