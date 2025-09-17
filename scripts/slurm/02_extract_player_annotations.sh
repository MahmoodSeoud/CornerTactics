#!/bin/bash
#SBATCH --job-name=unzip_frames
#SBATCH --output=logs/slurm/unzip_frames_%j.out
#SBATCH --error=logs/slurm/unzip_frames_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting to unzip Frames-v3.zip files at $(date)"
echo "Found $(find data/datasets/soccernet/soccernet_videos -name 'Frames-v3.zip' | wc -l) zip files"

# Unzip all Frames-v3.zip files into v3_frames subdirectory
find data/datasets/soccernet/soccernet_videos -name "Frames-v3.zip" | \
    parallel -j 8 'echo "Unzipping {}"; dir={//}/v3_frames; mkdir -p "$dir"; unzip -q -o {} -d "$dir"'

echo "Unzipping complete at $(date)"

# Optional: Remove zip files to save space (uncomment if needed)
# find data/datasets/soccernet/soccernet_videos -name "Frames-v3.zip" -delete