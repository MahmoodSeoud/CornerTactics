#!/bin/bash
#SBATCH --job-name=tacticai_baselines
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/train_baselines_%j.out
#SBATCH --error=logs/train_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# TacticAI Baseline Training Script
# Days 5-6: Train Random, XGBoost, and MLP baselines

echo "=========================================="
echo "TacticAI Baseline Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

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
