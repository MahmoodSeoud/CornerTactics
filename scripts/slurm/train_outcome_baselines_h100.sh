#!/bin/bash
#SBATCH --job-name=outcome_h100
#SBATCH --partition=acltr
#SBATCH --output=logs/outcome_baselines_h100_%j.out
#SBATCH --error=logs/outcome_baselines_h100_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100_80gb:1

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set paths
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install xgboost scikit-learn torch torch-geometric --quiet

echo "=========================================="
echo "Multi-Class Outcome Baselines on H100"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# Train all three baselines
echo "Training Random, XGBoost, and MLP outcome baselines..."
python scripts/training/train_outcome_baselines.py \
    --models all \
    --batch-size 128 \
    --num-steps 15000 \
    --device cuda \
    --output-dir results/baselines \
    --random-seed 42

echo ""
echo "=========================================="
echo "H100 Training Complete"
echo "=========================================="
echo "Results saved to: results/baselines/"
echo "Date: $(date)"
