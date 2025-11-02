#!/bin/bash
#SBATCH --job-name=baseline_a100
#SBATCH --partition=acltr
#SBATCH --output=logs/baselines_a100_%j.out
#SBATCH --error=logs/baselines_a100_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_40gb:1

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set paths
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install xgboost scikit-learn torch torch-geometric --quiet

echo "=========================================="
echo "Training Baseline Models on A100 40GB"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# First fix the receiver mapping
echo ""

echo ""
echo "Step 1: Training Random Baseline..."
python scripts/training/train_baselines.py \
    --model random \
    --gpu-type a100 \
    --batch-size 128

echo ""
echo "Step 2: Training XGBoost Baseline (Enhanced)..."
python scripts/training/train_baselines.py \
    --model xgboost \
    --gpu-type a100 \
    --batch-size 128 \
    --xgb-n-estimators 1000 \
    --xgb-max-depth 8 \
    --xgb-learning-rate 0.03

echo ""
echo "Step 3: Training MLP Baseline (Extended)..."
python scripts/training/train_baselines.py \
    --model mlp \
    --gpu-type a100 \
    --batch-size 128 \
    --mlp-steps 20000 \
    --mlp-lr 0.0005 \
    --mlp-weight-decay 0.00005 \
    --mlp-hidden1 512 \
    --mlp-hidden2 256 \
    --mlp-dropout 0.25 \
    --eval-every 1000 \
    --shot-weight 1.5 \
    --save-model

echo ""
echo "=========================================="
echo "A100 Training Complete"
echo "=========================================="
echo "Results saved to: results/baselines/"
echo "Date: $(date)"