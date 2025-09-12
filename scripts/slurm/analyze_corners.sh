#!/bin/bash
#SBATCH --job-name=corner_analysis
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/analysis_%j.log
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Activate conda environment
source ~/.bashrc
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner analysis at $(date)"
echo "Node: $(hostname)"

# Use research lab organized data structure
DATA_DIR="data/datasets/soccernet/soccernet_videos"
echo "Using data directory: $DATA_DIR"

# Create results directory
mkdir -p data/insights

# Run analysis only (no video extraction)
echo "Analyzing corners from $DATA_DIR..."
python main.py --data-dir $DATA_DIR --no-clips --output data/insights/corners_analysis.csv

echo "Corner analysis completed at $(date)"
echo "Results saved to data/insights/corners_analysis.csv"