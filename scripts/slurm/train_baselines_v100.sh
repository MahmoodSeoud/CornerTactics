#!/bin/bash
#SBATCH --job-name=baseline_v100
#SBATCH --partition=acltr
#SBATCH --output=logs/baselines_v100_%j.out
#SBATCH --error=logs/baselines_v100_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set paths
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install xgboost scikit-learn torch torch-geometric --quiet

echo "=========================================="
echo "Training Baseline Models on V100"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

echo "Step 1: Training Random Baseline..."
python scripts/training/train_baselines.py \
    --model random \
    --gpu-type v100 \
    --batch-size 64

echo ""
echo "Step 2: Training XGBoost Baseline..."
python scripts/training/train_baselines.py \
    --model xgboost \
    --gpu-type v100 \
    --batch-size 64 \
    --xgb-n-estimators 500 \
    --xgb-max-depth 6 \
    --xgb-learning-rate 0.05

echo ""
echo "Step 3: Training MLP Baseline..."
python scripts/training/train_baselines.py \
    --model mlp \
    --gpu-type v100 \
    --batch-size 64 \
    --mlp-steps 10000 \
    --mlp-lr 0.001 \
    --mlp-weight-decay 0.0001 \
    --mlp-hidden1 256 \
    --mlp-hidden2 128 \
    --mlp-dropout 0.3 \
    --eval-every 500 \
    --save-model

echo ""
echo "=========================================="
echo "V100 Training Complete"
echo "=========================================="
echo "Results saved to: results/baselines/"
echo "Date: $(date)"