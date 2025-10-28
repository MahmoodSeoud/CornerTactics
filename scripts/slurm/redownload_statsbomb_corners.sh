#!/bin/bash
#SBATCH --job-name=redownload_sb_corners
#SBATCH --partition=scavenge
#SBATCH --output=logs/redownload_sb_%j.out
#SBATCH --error=logs/redownload_sb_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Re-download StatsBomb corners with receiver information
# Updated download script now extracts:
# - Receiver name (pass_recipient)
# - Receiver location (from Ball Receipt event)

echo "============================================================"
echo "RE-DOWNLOADING STATSBOMB CORNERS WITH RECEIVER INFO"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Fix libstdc++ compatibility issue
export LD_LIBRARY_PATH=/opt/itu/easybuild/software/Anaconda3/2024.02-1/lib:$LD_LIBRARY_PATH

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install/update dependencies
echo "Installing dependencies..."
pip install statsbombpy pandas tqdm requests --quiet

# Create output directory
mkdir -p data/raw/statsbomb logs

echo ""
echo "Starting re-download with receiver extraction..."
echo ""

# Run the updated download script
python scripts/download_statsbomb_corners.py

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
df = pd.read_csv('data/raw/statsbomb/corners_360.csv')
print(f'Total corners: {len(df)}')

# Check receiver coverage
has_receiver_name = df['receiver_name'].notna().sum()
has_receiver_location = df['receiver_location_x'].notna().sum()

print(f'Corners with receiver name: {has_receiver_name} ({100*has_receiver_name/len(df):.1f}%)')
print(f'Corners with receiver location: {has_receiver_location} ({100*has_receiver_location/len(df):.1f}%)')
    " 2>/dev/null || echo "Could not compute statistics"

    echo ""
    echo "Next steps:"
    echo "1. Update add_receiver_labels.py to use Ball Receipt location matching"
    echo "2. Re-run receiver label mapping"
    echo "3. Remove hash workaround from ReceiverCornerDataset"
    echo "4. Re-test Day 3-4 implementation"
else
    echo ""
    echo "============================================================"
    echo "❌ Download Failed"
    echo "============================================================"
    echo "Check error log: logs/redownload_sb_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: ${SECONDS} seconds"
echo "============================================================"
