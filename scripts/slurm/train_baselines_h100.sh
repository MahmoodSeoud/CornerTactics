#!/bin/bash
#SBATCH --job-name=baseline_h100
#SBATCH --partition=acltr
#SBATCH --output=logs/baselines_h100_%j.out
#SBATCH --error=logs/baselines_h100_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set paths
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install xgboost scikit-learn torch torch-geometric --quiet

echo "=========================================="
echo "Training Baseline Models on H100"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# First fix the receiver mapping

echo ""
echo "Step 1: Training Random Baseline..."
python scripts/training/train_baselines.py \
    --model random \
    --gpu-type h100 \
    --batch-size 256

echo ""
echo "Step 2: Training XGBoost Baseline (Optimized)..."
python scripts/training/train_baselines.py \
    --model xgboost \
    --gpu-type h100 \
    --batch-size 256 \
    --xgb-n-estimators 1500 \
    --xgb-max-depth 10 \
    --xgb-learning-rate 0.02

echo ""
echo "Step 3: Training MLP Baseline (Deep)..."
python scripts/training/train_baselines.py \
    --model mlp \
    --gpu-type h100 \
    --batch-size 256 \
    --mlp-steps 30000 \
    --mlp-lr 0.0003 \
    --mlp-weight-decay 0.00001 \
    --mlp-hidden1 512 \
    --mlp-hidden2 256 \
    --mlp-dropout 0.2 \
    --eval-every 1500 \
    --shot-weight 2.0 \
    --save-model

echo ""
echo "=========================================="
echo "H100 Training Complete"
echo "=========================================="
echo "Results saved to: results/baselines/"
echo "Date: $(date)"