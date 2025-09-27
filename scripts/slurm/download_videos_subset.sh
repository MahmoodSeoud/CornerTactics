#!/bin/bash
#SBATCH --job-name=sn_videos
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sn_download_videos_%j.out
#SBATCH --error=logs/sn_download_videos_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# ==============================================================================
# Download Videos for Specific Games
# Large download - only get what we need for testing
# ==============================================================================

echo "Starting SoccerNet Videos download at $(date)"
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
echo "Target: Download videos for games that have camera labels"
echo "This ensures we have both videos + camera annotations"

# First check what games have camera labels
GAMES_WITH_CAMERAS=$(find data/datasets/soccernet -name "Labels-cameras.json" | wc -l)
echo "Found $GAMES_WITH_CAMERAS games with camera labels"

if [ "$GAMES_WITH_CAMERAS" -eq 0 ]; then
    echo "❌ No camera labels found. Run download_camera_labels.sh first"
    exit 1
fi

echo ""
echo "Downloading videos for subset of games (valid split only)..."

python src/download_soccernet.py \
    --videos-only \
    --password "$PASSWORD" \
    --data-dir data/datasets/soccernet \
    --splits valid

echo ""
echo "Download finished at $(date)"

# Verify downloads
echo ""
echo "Downloaded videos summary:"
VIDEO_COUNT=$(find data/datasets/soccernet -name "*.mkv" | wc -l)
echo "Video files: $VIDEO_COUNT"

if [ "$VIDEO_COUNT" -gt 0 ]; then
    echo "✅ Videos downloaded successfully"

    # Check for our target games
    echo ""
    echo "Sample downloaded videos:"
    find data/datasets/soccernet -name "*.mkv" | head -5 | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "  $(basename $(dirname "$file")): $SIZE"
    done

    # Check if Chelsea game video exists
    CHELSEA_VIDEO="data/datasets/soccernet/videos/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"
    if [ -f "$CHELSEA_VIDEO" ]; then
        echo "🎯 Target Chelsea game video: FOUND"
        echo "Size: $(du -h "$CHELSEA_VIDEO" | cut -f1)"
    else
        echo "⚠️  Target Chelsea game video: NOT FOUND"
    fi
else
    echo "❌ No videos downloaded"
    exit 1
fi

echo ""
echo "Videos ready for calibration pipeline!"
