#!/bin/bash
#SBATCH --job-name=download_more
#SBATCH --partition=scavenge
#SBATCH --output=logs/download_more_%j.out
#SBATCH --error=logs/download_more_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "============================================================"
echo "Downloading MORE StatsBomb Corners"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "============================================================"

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Just need basic packages - avoid torch issues
pip install statsbombpy pandas tqdm --user --quiet 2>/dev/null

echo ""
echo "Starting download..."
echo "This will test all 75 competitions for 360 data"
echo ""

python scripts/download_more_corners.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Download Complete!"
    echo "============================================================"

    # Show what we got
    if [ -f data/raw/statsbomb/corners_360_expanded.csv ]; then
        echo ""
        echo "New dataset created: corners_360_expanded.csv"
        wc -l data/raw/statsbomb/corners_360*.csv
    fi
else
    echo ""
    echo "❌ Download failed"
fi

echo ""
echo "Job completed at: $(date)"