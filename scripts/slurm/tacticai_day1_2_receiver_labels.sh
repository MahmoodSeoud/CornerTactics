#!/bin/bash
#SBATCH --job-name=receiver_labels
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/receiver_labels_%j.out
#SBATCH --error=logs/receiver_labels_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# TacticAI Day 1-2: Receiver Label Extraction
# Adds receiver_player_id to CornerGraph metadata

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "=== TacticAI Day 1-2: Receiver Label Extraction ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Install dependencies if needed
pip install tqdm --quiet

# Run receiver label extraction
python scripts/preprocessing/add_receiver_labels.py

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
