#!/bin/bash
#SBATCH --job-name=corner_extract
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/corners_%j.log
#SBATCH --error=logs/corners_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Load required modules
module load python/3.9

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner extraction at $(date)"
echo "Node: $(hostname)"

# Use standard data directory
DATA_DIR="data"
echo "Using data directory: $DATA_DIR"

# Run corner extraction
echo "Extracting corners from $DATA_DIR..."
python main.py --data-dir $DATA_DIR --output results/corners.csv

echo "Corner extraction completed at $(date)"