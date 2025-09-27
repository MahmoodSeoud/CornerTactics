#!/bin/bash
#SBATCH --job-name=sn_cameras
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sn_download_cameras_%j.out
#SBATCH --error=logs/sn_download_cameras_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# ==============================================================================
# Download Camera Shot Labels (Labels-cameras.json)
# For professional broadcast camera annotations
# ==============================================================================

echo "Starting SoccerNet Camera Labels download at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

# Create output directories
mkdir -p data/datasets/soccernet/videos
mkdir -p logs

echo ""
echo "Downloading camera shot annotations (Labels-cameras.json)..."

python -c "
from SoccerNet.Downloader import SoccerNetDownloader

# Initialize downloader
downloader = SoccerNetDownloader(LocalDirectory='data/datasets/soccernet/videos')

# Download camera labels for all splits
print('Downloading Labels-cameras.json for all splits...')
downloader.downloadGames(files=['Labels-cameras.json'], split=['train', 'valid', 'test'])

print('Camera labels download completed!')
"

echo ""
echo "Download finished at $(date)"

# Verify downloads
echo ""
echo "Downloaded camera labels summary:"
CAMERA_COUNT=$(find data/datasets/soccernet -name "Labels-cameras.json" | wc -l)
echo "Labels-cameras.json: $CAMERA_COUNT files"

if [ "$CAMERA_COUNT" -gt 0 ]; then
    echo "✅ Camera labels downloaded successfully"

    # Show sample of what we got
    echo ""
    echo "Sample camera label files:"
    find data/datasets/soccernet -name "Labels-cameras.json" | head -5

    # Check if our target Chelsea game has camera labels
    CHELSEA_CAMERAS="data/datasets/soccernet/videos/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-cameras.json"
    if [ -f "$CHELSEA_CAMERAS" ]; then
        echo "🎯 Target Chelsea game camera labels: FOUND"
    else
        echo "⚠️  Target Chelsea game camera labels: NOT FOUND"
        echo "Available Chelsea games with camera labels:"
        find data/datasets/soccernet -name "Labels-cameras.json" | grep -i chelsea | head -3
    fi
else
    echo "❌ No camera labels downloaded"
    exit 1
fi

echo ""
echo "Camera labels ready for calibration pipeline!"
