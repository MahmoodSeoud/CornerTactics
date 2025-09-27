#!/bin/bash
#SBATCH --job-name=sn_corners
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sn_corners_%j.out
#SBATCH --error=logs/sn_corners_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

# Add project root to Python path for src module imports
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "Starting corner frame extraction at $(date)"
echo "Node: $(hostname)"

# Create output directories
mkdir -p logs
mkdir -p data/insights

# Extract 20-second video clips around corner kick moments
echo "Extracting 20-second video clips around corner kick timestamps..."
python scripts/extract_corners.py \
    --data-dir data \
    --output data/insights/corner_clips_metadata.csv \
    --split-by-visibility

echo "Corner clip extraction completed at $(date)"

# Report statistics
echo ""
echo "Extraction results:"
if [ -f "data/insights/corner_clips_metadata.csv" ]; then
    echo "Metadata CSV created: data/insights/corner_clips_metadata.csv"
    echo "Total corners processed: $(tail -n +2 data/insights/corner_clips_metadata.csv | wc -l)"
    echo "Successful extractions: $(tail -n +2 data/insights/corner_clips_metadata.csv | grep -v ',,$' | wc -l)"
    echo "Failed extractions: $(tail -n +2 data/insights/corner_clips_metadata.csv | grep ',,$' | wc -l)"
else
    echo "⚠ CSV file not created - check for errors above"
fi

if [ -d "data/datasets/soccernet/corner_clips" ]; then
    echo "Clip directories created:"
    echo "  Visible clips: data/datasets/soccernet/corner_clips/visible"
    echo "  Not shown clips: data/datasets/soccernet/corner_clips/not_shown"

    if [ -d "data/datasets/soccernet/corner_clips/visible" ]; then
        echo "Visible clips extracted: $(find data/datasets/soccernet/corner_clips/visible -name "*.mp4" | wc -l)"
        echo "Visible storage: $(du -sh data/datasets/soccernet/corner_clips/visible | cut -f1)"
    fi

    if [ -d "data/datasets/soccernet/corner_clips/not_shown" ]; then
        echo "Not shown clips extracted: $(find data/datasets/soccernet/corner_clips/not_shown -name "*.mp4" | wc -l)"
        echo "Not shown storage: $(du -sh data/datasets/soccernet/corner_clips/not_shown | cut -f1)"
    fi

    echo "Total clips extracted: $(find data/datasets/soccernet/corner_clips -name "*.mp4" | wc -l)"
    echo "Total storage used: $(du -sh data/datasets/soccernet/corner_clips | cut -f1)"
else
    echo "⚠ Clip directory not found - check for errors above"
fi