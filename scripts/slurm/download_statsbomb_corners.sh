#!/bin/bash
#SBATCH --job-name=sb_corners
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sb_corners_%j.out
#SBATCH --error=logs/sb_corners_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# StatsBomb Corner Download
# Downloads all professional men's corner kicks with 360 player positions
# Excludes: women's competitions, youth competitions

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "Starting StatsBomb 360 fast download at $(date)"
echo "Node: $(hostname)"
echo ""

# Install dependencies if needed
pip install statsbombpy tqdm pandas --quiet

# Run download script
python scripts/download_statsbomb_corners.py

echo ""
echo "Download completed at $(date)"

# Show results
if [ -f "data/statsbomb/corners_360.csv" ]; then
    echo ""
    echo "✓ Success!"
    echo "  Corners CSV: data/statsbomb/corners_360.csv"
    echo "  Lines: $(wc -l < data/statsbomb/corners_360.csv)"
    echo "  Size: $(du -h data/statsbomb/corners_360.csv | cut -f1)"
else
    echo "✗ Error: Output file not created"
    exit 1
fi
