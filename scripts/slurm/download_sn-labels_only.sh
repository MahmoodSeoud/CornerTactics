#!/bin/bash
#SBATCH --job-name=sn_labels
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sn_download_labels_%j.out
#SBATCH --error=logs/sn_download_labels_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# ==============================================================================
# Download Event Labels Only (Labels-v2.json and Labels-v3.json)
# Fast download for corner event timestamps
# ==============================================================================

echo "Starting SoccerNet Event Labels download at $(date)"
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
echo "Downloading event labels (Labels-v2.json and Labels-v3.json)..."

python src/download_soccernet.py \
    --labels-only \
    --password "$PASSWORD" \
    --data-dir data/datasets/soccernet \
    --splits train valid test

echo ""
echo "Download finished at $(date)"

# Verify downloads
echo ""
echo "Downloaded labels summary:"
LABELS_V2_COUNT=$(find data/datasets/soccernet -name "Labels-v2.json" | wc -l)
LABELS_V3_COUNT=$(find data/datasets/soccernet -name "Labels-v3.json" | wc -l)

echo "Labels-v2.json: $LABELS_V2_COUNT files"
echo "Labels-v3.json: $LABELS_V3_COUNT files"

if [ "$LABELS_V2_COUNT" -gt 0 ]; then
    echo "✅ Event labels downloaded successfully"

    # Show sample of what we got
    echo ""
    echo "Sample games with labels:"
    find data/datasets/soccernet -name "Labels-v2.json" | head -5 | while read file; do
        echo "  $(dirname "$file" | sed 's|.*/||')"
    done
else
    echo "❌ No event labels downloaded"
    exit 1
fi

echo ""
echo "Event labels ready for corner frame extraction!"
