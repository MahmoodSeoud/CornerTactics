#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --output=logs/baseline_train_%j.out
#SBATCH --error=logs/baseline_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set working directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install torch_geometric scikit-learn --quiet

# Run baseline training
echo "Starting baseline training..."
python scripts/training/train_baseline.py \
    --graph-path data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl \
    --batch-size 32 \
    --num-steps 10000 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --device cuda \
    --random-state 42 \
    --output-dir results

echo "Baseline training complete!"
