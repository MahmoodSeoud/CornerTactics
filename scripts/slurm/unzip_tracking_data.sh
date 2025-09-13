#!/bin/bash
#SBATCH --job-name=unzip_tracking
#SBATCH --output=logs/slurm/unzip_tracking_%j.out
#SBATCH --error=logs/slurm/unzip_tracking_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cores

# Create log directory if it doesn't exist
mkdir -p logs/slurm

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robo

echo "Starting tracking data extraction at $(date)"
echo "========================================="

# Define paths
TRACKING_DIR="data/datasets/soccernet/soccernet_tracking"
cd $TRACKING_DIR

# Function to unzip with progress reporting
unzip_with_progress() {
    local zipfile=$1
    local target_dir=$2
    
    echo "Extracting $zipfile to $target_dir..."
    echo "File size: $(du -h $zipfile | cut -f1)"
    
    # Create target directory
    mkdir -p $target_dir
    
    # Unzip with verbose output for progress
    unzip -o $zipfile -d $target_dir
    
    # Report completion
    echo "Completed extracting $zipfile"
    echo "Extracted size: $(du -sh $target_dir | cut -f1)"
    echo "Number of files: $(find $target_dir -type f | wc -l)"
    echo "----------------------------------------"
}

# Extract train data
if [ -f "tracking/train.zip" ]; then
    echo "Processing train.zip (9.6GB compressed)..."
    unzip_with_progress "tracking/train.zip" "train"
else
    echo "WARNING: train.zip not found!"
fi

# Extract test data
if [ -f "tracking/test.zip" ]; then
    echo "Processing test.zip (8.7GB compressed)..."
    unzip_with_progress "tracking/test.zip" "test"
else
    echo "WARNING: test.zip not found!"
fi

# Extract challenge data
if [ -f "tracking/challenge.zip" ]; then
    echo "Processing challenge.zip (11GB compressed)..."
    unzip_with_progress "tracking/challenge.zip" "challenge"
else
    echo "WARNING: challenge.zip not found!"
fi

echo "========================================="
echo "Extraction complete at $(date)"

# Summary statistics
echo ""
echo "Final directory structure:"
ls -la .

echo ""
echo "Disk usage summary:"
du -sh train test challenge 2>/dev/null || echo "Some directories not found"

echo ""
echo "Sample tracking data structure (first sequence):"
if [ -d "train" ]; then
    first_seq=$(ls train | head -1)
    if [ -n "$first_seq" ]; then
        echo "Sample from train/$first_seq:"
        ls -la "train/$first_seq" | head -10
        
        # Show sample of tracking data format
        if [ -f "train/$first_seq/gt/gt.txt" ]; then
            echo ""
            echo "Sample ground truth data (first 5 lines):"
            head -5 "train/$first_seq/gt/gt.txt"
        fi
        
        if [ -f "train/$first_seq/gameinfo.ini" ]; then
            echo ""
            echo "Game info:"
            cat "train/$first_seq/gameinfo.ini"
        fi
    fi
fi

echo ""
echo "Job completed successfully!"