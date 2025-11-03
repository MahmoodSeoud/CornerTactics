#!/bin/bash
#SBATCH --job-name=baseline_v2
#SBATCH --partition=scavenge
#SBATCH --output=logs/baselines_v2_%j.out
#SBATCH --error=logs/baselines_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# Train Baseline Models with v2 Receiver Labels (100% coverage)
# Dataset: 5,814 graphs (up from 3,492) with event-stream receiver labeling

echo "=========================================="
echo "Training Baselines with v2 Dataset"
echo "=========================================="
echo "Dataset: statsbomb_temporal_augmented_with_receiver_v2.pkl"
echo "Coverage: 100% (5,814/5,814 graphs)"
echo "New features: +37.3% defensive receivers"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set paths
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install xgboost scikit-learn torch --quiet

# Data path
DATA_PATH="data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl"

echo ""
echo "=========================================="
echo "Step 1/3: Training Random Baseline"
echo "=========================================="
python scripts/training/train_baselines.py \
    --model random \
    --data-path $DATA_PATH \
    --batch-size 128 \
    --gpu-type v2

echo ""
echo "=========================================="
echo "Step 2/3: Training XGBoost Baseline"
echo "=========================================="
python scripts/training/train_baselines.py \
    --model xgboost \
    --data-path $DATA_PATH \
    --batch-size 128 \
    --xgb-n-estimators 1000 \
    --xgb-max-depth 8 \
    --xgb-learning-rate 0.03 \
    --gpu-type v2

echo ""
echo "=========================================="
echo "Step 3/3: Training MLP Baseline"
echo "=========================================="
python scripts/training/train_baselines.py \
    --model mlp \
    --data-path $DATA_PATH \
    --batch-size 128 \
    --mlp-steps 20000 \
    --mlp-lr 0.0005 \
    --mlp-weight-decay 0.00005 \
    --mlp-hidden1 512 \
    --mlp-hidden2 256 \
    --mlp-dropout 0.25 \
    --eval-every 1000 \
    --shot-weight 1.5 \
    --save-model \
    --gpu-type v2

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: results/baselines/"
echo "Date: $(date)"
echo ""
echo "Comparison:"
echo "  Old dataset: 3,492 graphs (60.1% coverage)"
echo "  New dataset: 5,814 graphs (100% coverage)"
echo "  Improvement: +2,322 graphs (+66.5%)"
