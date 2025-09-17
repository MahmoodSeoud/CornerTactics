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

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner extraction at $(date)"
echo "Node: $(hostname)"

# Use research lab organized data structure
DATA_DIR="data/datasets/soccernet/soccernet_videos"
echo "Using data directory: $DATA_DIR"

# Run corner extraction
echo "Extracting corners from $DATA_DIR..."
python main.py --data-dir $DATA_DIR --output data/insights/corners_with_clips.csv

echo "Corner extraction completed at $(date)"