#!/bin/bash
#SBATCH --job-name=soccernet_analysis
#SBATCH --output=logs/slurm/soccernet_analysis_%j.out
#SBATCH --error=logs/slurm/soccernet_analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics/SoccerNet-v3

echo "Starting SoccerNet analysis at $(date)"

# Create output directories
mkdir -p ../data/insights/statistics
mkdir -p ../data/insights/visualizations

# Run statistics generation
echo "Generating statistics..."
python statistics.py \
    --SoccerNet_path ../data/datasets/soccernet/soccernet_videos \
    --save_path ../data/insights/statistics/ \
    --split all \
    --num_workers 8 \
    --resolution_width 1920 \
    --resolution_height 1080

# Run visualization
echo "Creating visualizations..."
python visualize.py \
    --SoccerNet_path ../data/datasets/soccernet/soccernet_videos \
    --save_path ../data/insights/visualizations/ \
    --split all \
    --num_workers 8 \
    --resolution_width 1920 \
    --resolution_height 1080 \
    --tiny 10

echo "Analysis complete at $(date)"