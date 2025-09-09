#!/bin/bash
#SBATCH --job-name=corner_analysis
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/analysis_%j.log
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load required modules
module load python/3.9

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner analysis at $(date)"
echo "Node: $(hostname)"

# Use standard data directory
DATA_DIR="data"
echo "Using data directory: $DATA_DIR"

# Create results directory
mkdir -p results

# Run analysis only (no video extraction)
echo "Analyzing corners from $DATA_DIR..."
python main.py --data-dir $DATA_DIR --no-clips --output results/analysis.csv

echo "Corner analysis completed at $(date)"
echo "Results saved to results/analysis.csv"