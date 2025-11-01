#!/bin/bash
#SBATCH --job-name=tacticai_baselines_h100
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/train_baselines_h100_%j.out
#SBATCH --error=logs/train_baselines_h100_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# TacticAI Baseline Training Script - H100 GPU
# Days 5-6: Train Random, XGBoost, and MLP baselines with proper dual-task learning

echo "=========================================="
echo "TacticAI Baseline Training - H100 GPU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: H100"
echo "Start Time: $(date)"
echo "=========================================="

# Display GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set working directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install XGBoost if not already installed
pip install xgboost --quiet

# Run training script
python scripts/training/train_baseline.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="