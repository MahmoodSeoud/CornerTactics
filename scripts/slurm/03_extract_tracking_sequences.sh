#!/bin/bash
#SBATCH --job-name=unzip_tracking
#SBATCH --output=logs/slurm/unzip_tracking_%j.out
#SBATCH --error=logs/slurm/unzip_tracking_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting to unzip tracking data at $(date)"

# Create directories for extracted data
mkdir -p data/datasets/soccernet/soccernet_tracking/train
mkdir -p data/datasets/soccernet/soccernet_tracking/test

# Unzip train.zip
echo "Unzipping train.zip (9.5GB)..."
unzip -q -o data/datasets/soccernet/soccernet_tracking/tracking/train.zip \
    -d data/datasets/soccernet/soccernet_tracking/

# Unzip test.zip
echo "Unzipping test.zip (8.7GB)..."
unzip -q -o data/datasets/soccernet/soccernet_tracking/tracking/test.zip \
    -d data/datasets/soccernet/soccernet_tracking/

echo "Unzipping complete at $(date)"

# List extracted contents
echo "Extracted contents:"
ls -la data/datasets/soccernet/soccernet_tracking/train/ | head -20
ls -la data/datasets/soccernet/soccernet_tracking/test/ | head -20

# Optional: Remove zip files to save space (uncomment if needed)
# rm data/datasets/soccernet/soccernet_tracking/tracking/*.zip