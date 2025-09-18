#!/bin/bash
#SBATCH --job-name=unzip_data
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/slurm/unzip_%j.out
#SBATCH --error=logs/slurm/unzip_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

echo "Starting data extraction at $(date)"
echo "Node: $(hostname)"

DATA_DIR="data/datasets/soccernet"

# Extract tracking data if zipped
echo "Checking for tracking data zips..."
for split in train test challenge; do
    ZIP_FILE="${DATA_DIR}/tracking/${split}.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "Extracting ${split}.zip..."
        unzip -q -o "$ZIP_FILE" -d "${DATA_DIR}/tracking/"
        echo "✓ Extracted ${split} tracking data"
    fi
done

# The SoccerNet downloader usually extracts data automatically, but check for any other zips
echo ""
echo "Checking for other compressed files..."
find "$DATA_DIR" -name "*.zip" -o -name "*.tar.gz" | while read archive; do
    dir=$(dirname "$archive")
    echo "Found: $archive"
    echo "  Extracting to: $dir"

    if [[ "$archive" == *.zip ]]; then
        unzip -q -o "$archive" -d "$dir"
    elif [[ "$archive" == *.tar.gz ]]; then
        tar -xzf "$archive" -C "$dir"
    fi

    echo "  ✓ Extracted $(basename "$archive")"
done

echo ""
echo "Data extraction completed at $(date)"

# Report final structure
echo ""
echo "Final data structure:"
echo "Labels: $(find "$DATA_DIR" -name "Labels-v3.json" | wc -l) files"
echo "Videos: $(find "$DATA_DIR" -name "*.mkv" | wc -l) files"
echo "Tracking sequences: $(find "$DATA_DIR" -type d -name "SNMOT-*" | wc -l)"