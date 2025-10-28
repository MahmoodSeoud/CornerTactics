#!/bin/bash
#SBATCH --job-name=download_all_sb
#SBATCH --partition=scavenge
#SBATCH --output=logs/download_all_sb_%j.out
#SBATCH --error=logs/download_all_sb_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Download ALL corners from ALL StatsBomb competitions
# Expected: 5,000-10,000 corners (vs current 1,118)

echo "============================================================"
echo "COMPREHENSIVE STATSBOMB DATA DOWNLOAD"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install/update dependencies
echo "Installing dependencies..."
pip install statsbombpy pandas tqdm requests --quiet

# Create output directory
mkdir -p data/raw/statsbomb logs

echo ""
echo "Starting comprehensive download from ALL competitions..."
echo "This will take several hours as we download from 75+ competitions"
echo ""

# Run the download script
python scripts/download_all_statsbomb_data.py \
    --output data/raw/statsbomb/all_corners_360.csv \
    --resume

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Download Complete"
    echo "============================================================"

    # Show statistics
    echo ""
    echo "Dataset statistics:"
    python -c "
import pandas as pd
df = pd.read_csv('data/raw/statsbomb/all_corners_360.csv')
print(f'Total corners: {len(df)}')
print(f'Competitions: {df[\"competition\"].nunique()}')
print(f'Expected with augmentation: {len(df) * 10:,} samples')
    " 2>/dev/null || echo "Could not compute statistics"

    echo ""
    echo "Next steps:"
    echo "1. Process corners through feature extraction pipeline"
    echo "2. Build graphs with expanded dataset"
    echo "3. Train model with 10× more data"
else
    echo ""
    echo "============================================================"
    echo "❌ Download Failed"
    echo "============================================================"
    echo "Check error log: logs/download_all_sb_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: ${SECONDS} seconds"
echo "============================================================"