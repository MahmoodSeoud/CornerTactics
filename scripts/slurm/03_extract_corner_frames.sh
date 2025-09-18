#!/bin/bash
#SBATCH --job-name=corner_frames
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/slurm/corner_frames_%j.out
#SBATCH --error=logs/slurm/corner_frames_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting corner frame extraction at $(date)"
echo "Node: $(hostname)"

# Create output directories
mkdir -p logs/slurm
mkdir -p data/insights

# Extract single frames at corner kick moments using new CLI
echo "Extracting single frames at corner kick timestamps..."
python src/cli.py \
    --data-dir data \
    --output data/insights/corner_frames_metadata.csv

echo "Corner frame extraction completed at $(date)"

# Report statistics
echo ""
echo "Extraction results:"
if [ -f "data/insights/corner_frames_metadata.csv" ]; then
    echo "Metadata CSV created: data/insights/corner_frames_metadata.csv"
    echo "Total corners processed: $(tail -n +2 data/insights/corner_frames_metadata.csv | wc -l)"
    echo "Successful extractions: $(tail -n +2 data/insights/corner_frames_metadata.csv | grep -v ',,$' | wc -l)"
    echo "Failed extractions: $(tail -n +2 data/insights/corner_frames_metadata.csv | grep ',,$' | wc -l)"
else
    echo "⚠ CSV file not created - check for errors above"
fi

if [ -d "data/datasets/soccernet/soccernet_corner_frames" ]; then
    echo "Frame directory: data/datasets/soccernet/soccernet_corner_frames"
    echo "Total frames extracted: $(find data/datasets/soccernet/soccernet_corner_frames -name "*.jpg" | wc -l)"
    echo "Storage used: $(du -sh data/datasets/soccernet/soccernet_corner_frames | cut -f1)"
else
    echo "⚠ Frame directory not found - check for errors above"
fi