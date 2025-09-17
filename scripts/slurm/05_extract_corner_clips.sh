#!/bin/bash
#SBATCH --job-name=corner_frames
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/slurm/corner_frames_%j.out
#SBATCH --error=logs/slurm/corner_frames_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner frame extraction at $(date)"
echo "Node: $(hostname)"

# Use research lab organized data structure
echo "Using data directory: data"
echo "Videos are in: data/datasets/soccernet/soccernet_videos/"
echo "Frames will be saved to: data/datasets/soccernet/soccernet_corner_frames/"

# Run corner frame extraction (single frames instead of clips)
echo "Extracting corner frames from all games (using Labels-v2.json)..."
python main.py --data-dir data --output data/insights/corners_with_frames_v2.csv --frames-only

echo "Corner frame extraction completed at $(date)"