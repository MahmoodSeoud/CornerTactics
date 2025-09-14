#!/bin/bash
#SBATCH --job-name=unzip_tracking
#SBATCH --output=logs/unzip_tracking_%j.log
#SBATCH --error=logs/unzip_tracking_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cores

# Activate conda environment (using same method as working scripts)
source ~/.bashrc
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting tracking data extraction at $(date)"
echo "========================================="

# Define paths
TRACKING_DIR="data/datasets/soccernet/soccernet_tracking"

# Function to unzip with progress reporting
unzip_with_progress() {
    local zipfile=$1
    local target_dir=$2
    
    echo "Extracting $zipfile..."
    echo "File size: $(du -h $TRACKING_DIR/$zipfile | cut -f1)"
    
    # Remove existing directory if needed
    if [ -d "$TRACKING_DIR/$target_dir" ]; then
        echo "Removing existing $target_dir directory..."
        rm -rf "$TRACKING_DIR/$target_dir"
    fi
    
    # Unzip directly (the zip already contains the folder structure)
    cd $TRACKING_DIR
    unzip -o $zipfile
    cd /home/mseo/CornerTactics
    
    # Report completion
    echo "Completed extracting $zipfile"
    echo "Extracted size: $(du -sh $TRACKING_DIR/$target_dir | cut -f1)"
    echo "Number of files: $(find $TRACKING_DIR/$target_dir -type f | wc -l)"
    echo "----------------------------------------"
}

# Check if zip files exist
echo "Checking for tracking data zip files..."
ls -lh $TRACKING_DIR/*.zip 2>/dev/null || echo "No zip files found in $TRACKING_DIR"

# Extract train data
if [ -f "$TRACKING_DIR/train.zip" ]; then
    echo "Processing train.zip (9.6GB compressed)..."
    unzip_with_progress "train.zip" "train"
else
    echo "WARNING: $TRACKING_DIR/train.zip not found!"
fi

# Extract test data
if [ -f "$TRACKING_DIR/test.zip" ]; then
    echo "Processing test.zip (8.7GB compressed)..."
    unzip_with_progress "test.zip" "test"
else
    echo "WARNING: $TRACKING_DIR/test.zip not found!"
fi

# Extract challenge data
if [ -f "$TRACKING_DIR/challenge.zip" ]; then
    echo "Processing challenge.zip (11GB compressed)..."
    unzip_with_progress "challenge.zip" "challenge"
else
    echo "WARNING: $TRACKING_DIR/challenge.zip not found!"
fi

echo "========================================="
echo "Extraction complete at $(date)"

# Summary statistics
echo ""
echo "Final directory structure:"
ls -la $TRACKING_DIR

echo ""
echo "Disk usage summary:"
du -sh $TRACKING_DIR/train $TRACKING_DIR/test $TRACKING_DIR/challenge 2>/dev/null || echo "Some directories not found"

echo ""
echo "Sample tracking data structure (first sequence):"
if [ -d "$TRACKING_DIR/train" ]; then
    first_seq=$(ls $TRACKING_DIR/train | head -1)
    if [ -n "$first_seq" ]; then
        echo "Sample from train/$first_seq:"
        ls -la "$TRACKING_DIR/train/$first_seq" | head -10
        
        # Show sample of tracking data format
        if [ -f "$TRACKING_DIR/train/$first_seq/gt/gt.txt" ]; then
            echo ""
            echo "Sample ground truth data (first 5 lines):"
            head -5 "$TRACKING_DIR/train/$first_seq/gt/gt.txt"
        fi
        
        if [ -f "$TRACKING_DIR/train/$first_seq/gameinfo.ini" ]; then
            echo ""
            echo "Game info:"
            cat "$TRACKING_DIR/train/$first_seq/gameinfo.ini"
        fi
    fi
fi

echo ""
echo "Job completed successfully!"