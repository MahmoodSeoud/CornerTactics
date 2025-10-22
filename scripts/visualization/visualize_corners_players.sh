#!/bin/bash
#SBATCH --job-name=viz_360
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/viz_360_%j.out
#SBATCH --error=logs/viz_360_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "Starting 360 visualization at $(date)"
echo "Node: $(hostname)"

# Install dependencies
pip install requests mplsoccer --quiet

python scripts/visualize_corners_with_players.py

echo ""
echo "Visualization completed at $(date)"

if [ -f "data/statsbomb/corners_with_players_2x2.png" ]; then
    echo "✓ Output: data/statsbomb/corners_with_players_2x2.png"
    FILE_SIZE=$(du -h data/statsbomb/corners_with_players_2x2.png | cut -f1)
    echo "  Size: $FILE_SIZE"
else
    echo "✗ Output file not created"
fi
