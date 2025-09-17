#!/bin/bash
#SBATCH --job-name=download_all_soccernet
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/slurm/download_all_%j.out
#SBATCH --error=logs/slurm/download_all_%j.err
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
mkdir -p data/datasets/soccernet/soccernet_videos
mkdir -p data/datasets/soccernet/soccernet_tracking

# SoccerNet password
PASSWORD="s0cc3rn3t"

echo "Starting complete SoccerNet download at $(date)"
echo "Node: $(hostname)"

# Test network speed first
echo "Testing network speed..."
curl -s https://www.google.com -o /dev/null -w "Speed: %{speed_download} bytes/s\n"

# Download both v2 and v3 labels (v2 has more corner annotations)
echo "Downloading v2 and v3 labels..."
python src/download_soccernet.py \
    --labels both \
    --data-dir data/datasets/soccernet/soccernet_videos \
    --splits train valid test

# Download v3 frames with bounding boxes (for player detection)
echo "Downloading v3 frames with bounding boxes..."
python src/download_soccernet.py \
    --frames v3 \
    --data-dir data/datasets/soccernet/soccernet_videos \
    --splits train valid test

# Download 720p broadcast videos (large files)
echo "Downloading 720p broadcast videos..."
python src/download_soccernet.py \
    --videos 720p \
    --password "$PASSWORD" \
    --data-dir data/datasets/soccernet/soccernet_videos \
    --splits train valid test

# Download tracking data
echo "Downloading tracking data..."
python src/download_soccernet.py \
    --tracklets tracking \
    --data-dir data/datasets/soccernet/soccernet_tracking \
    --splits train valid test

echo "Complete SoccerNet download finished at $(date)"