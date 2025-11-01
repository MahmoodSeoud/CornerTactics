#!/bin/bash
#SBATCH --job-name=config2_a100
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --output=logs/config2_a100_%j.out
#SBATCH --error=logs/config2_a100_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12

echo "=========================================="
echo "EXPERIMENT CONFIG 2: Weighted BCE Loss"
echo "GPU: A100 40GB"
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

python scripts/training/train_baseline_config2.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="