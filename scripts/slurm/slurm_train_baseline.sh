#!/bin/bash
#SBATCH --job-name=train_baseline_49
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --partition=scavenge

# Load required modules
module load GCC/14.2.0
module load CUDA/12.1.1

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library path for pandas compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run training pipeline
echo "Starting retraining pipeline with 49 features..."
echo "Start time: $(date)"

# Step 1: Create splits
echo "Step 1: Creating train/val/test splits..."
python scripts/04_create_splits.py

# Step 2: Train baseline models (Random Forest, XGBoost, MLP)
echo "Step 2: Training baseline models (4-class outcome prediction)..."
python scripts/05_train_baseline_models.py

# Step 3: Train binary shot models
echo "Step 3: Training binary shot prediction models..."
python scripts/08_train_binary_models.py

echo "Training complete!"
echo "End time: $(date)"

# List generated models
echo "Generated models:"
ls -lh models/*.pkl models/binary/*.pkl 2>/dev/null || echo "No models found"
