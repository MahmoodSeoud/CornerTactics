#!/bin/bash
#SBATCH --job-name=sn_tracking
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sn_download_tracking_%j.out
#SBATCH --error=logs/sn_download_tracking_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# ==============================================================================
# Download Tracking Data (SNMOT sequences)
# For validation and ground truth comparison
# ==============================================================================

echo "Starting SoccerNet Tracking Data download at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

# Create output directories
mkdir -p data/datasets/soccernet
mkdir -p logs

# SoccerNet password
PASSWORD="s0cc3rn3t"

echo ""
echo "Downloading tracking sequences (SNMOT data)..."

python src/download_soccernet.py \
    --tracking-only \
    --password "$PASSWORD" \
    --data-dir data/datasets/soccernet \
    --splits train valid test

echo ""
echo "Download finished at $(date)"

# Verify downloads
echo ""
echo "Downloaded tracking data summary:"
TRACKING_COUNT=$(find data/datasets/soccernet -name "gameinfo.ini" | wc -l)
echo "Tracking sequences: $TRACKING_COUNT"

if [ "$TRACKING_COUNT" -gt 0 ]; then
    echo "✅ Tracking data downloaded successfully"

    # Show sample of what we got
    echo ""
    echo "Sample tracking sequences:"
    find data/datasets/soccernet -name "gameinfo.ini" | head -5 | while read file; do
        echo "  $(dirname "$file" | sed 's|.*/||')"
    done

    # Check for corner sequences
    echo ""
    echo "Looking for corner sequences in tracking data..."
    CORNER_SEQUENCES=$(find data/datasets/soccernet -name "gameinfo.ini" -exec grep -l "Corner" {} \; | wc -l)
    echo "Corner sequences found: $CORNER_SEQUENCES"

    if [ "$CORNER_SEQUENCES" -gt 0 ]; then
        echo "Sample corner sequences:"
        find data/datasets/soccernet -name "gameinfo.ini" -exec grep -l "Corner" {} \; | head -3 | while read file; do
            echo "  $(dirname "$file" | sed 's|.*/||')"
        done
    fi
else
    echo "❌ No tracking data downloaded"
    exit 1
fi

echo ""
echo "Tracking data ready for validation!"
