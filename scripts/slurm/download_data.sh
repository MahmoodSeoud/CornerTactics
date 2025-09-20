#!/bin/bash
#SBATCH --job-name=download_soccernet
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

# Create output directories
mkdir -p logs/slurm
mkdir -p data/datasets/soccernet

# SoccerNet password
PASSWORD="s0cc3rn3t"

echo "Starting SoccerNet download at $(date)"
echo "Node: $(hostname)"

# Download all data needed for corner analysis
echo "Downloading all data (labels + tracking + videos)..."
python src/download_soccernet.py \
    --all \
    --password "$PASSWORD" \
    --data-dir data/datasets/soccernet \
    --splits train valid test

echo "Download finished at $(date)"

# Verify downloads
echo ""
echo "Downloaded data summary:"
echo "Labels-v2: $(find data/datasets/soccernet -name "Labels-v2.json" | wc -l) files"
echo "Labels-v3: $(find data/datasets/soccernet -name "Labels-v3.json" | wc -l) files"
echo "Videos: $(find data/datasets/soccernet -name "*.mkv" | wc -l) files"
echo "Tracking: $(find data/datasets/soccernet -name "gameinfo.ini" | wc -l) sequences"