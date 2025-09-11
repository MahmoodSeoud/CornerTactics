#!/bin/bash
#SBATCH --job-name=extract_tracklets
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/extract_tracklets_%j.log
#SBATCH --error=logs/extract_tracklets_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Load required modules
module load python/3.9

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting tracklet extraction at $(date)"
echo "Node: $(hostname)"

# Use research lab organized data structure
DATA_DIR="data/raw/soccernet_videos_720p"
TRACKING_DIR="$DATA_DIR/tracking"
OUTPUT_DIR="data/processed/tracking_extracted"

echo "Extracting tracklet ZIP files from $TRACKING_DIR"
echo "================================================"

# Check if tracking directory exists
if [ ! -d "$TRACKING_DIR" ]; then
    echo "ERROR: Tracking directory not found: $TRACKING_DIR"
    exit 1
fi

cd "$TRACKING_DIR"

# Create output directory
mkdir -p "../../processed/tracking_extracted"

# Extract train.zip if it exists
if [ -f "train.zip" ]; then
    echo "Extracting train.zip..."
    echo "File size: $(du -h train.zip)"
    unzip -q train.zip
    echo "✅ train.zip extracted successfully"
    echo "Extracted contents:"
    ls -la train/ | head -10
else
    echo "❌ train.zip not found"
fi

# Extract test.zip if it exists
if [ -f "test.zip" ]; then
    echo ""
    echo "Extracting test.zip..."
    echo "File size: $(du -h test.zip)"
    unzip -q test.zip
    echo "✅ test.zip extracted successfully"
    echo "Extracted contents:"
    ls -la test/ | head -10
else
    echo "❌ test.zip not found"
fi

# Extract challenge.zip if it exists
if [ -f "challenge.zip" ]; then
    echo ""
    echo "Extracting challenge.zip..."
    echo "File size: $(du -h challenge.zip)"
    unzip -q challenge.zip
    echo "✅ challenge.zip extracted successfully"
    echo "Extracted contents:"
    ls -la challenge/ | head -10
else
    echo "❌ challenge.zip not found"
fi

echo ""
echo "Summary of extracted tracklet data:"
echo "==================================="
find . -name "SNMOT-*" -type d | wc -l | sed 's/^/Total SNMOT sequences: /'
find . -name "gameinfo.ini" | wc -l | sed 's/^/Total gameinfo files: /'
find . -name "gt.txt" | wc -l | sed 's/^/Total ground truth files: /'

echo ""
echo "Disk usage after extraction:"
du -sh .

echo ""
echo "Tracklet extraction completed at $(date)"