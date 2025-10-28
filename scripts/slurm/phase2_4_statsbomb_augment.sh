#!/bin/bash
#SBATCH --job-name=phase2_4_sb_augment
#SBATCH --partition=acltr
#SBATCH --account=researchers
#SBATCH --output=logs/phase2_4_sb_augment_%j.out
#SBATCH --error=logs/phase2_4_sb_augment_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Phase 2.4: StatsBomb Temporal Augmentation
# Applies US Soccer Federation temporal augmentation approach
# Creates 5 temporal frames + mirror augmentation per corner
# Uses position perturbations to simulate movement uncertainty

echo "========================================"
echo "Phase 2.4: StatsBomb Temporal Augmentation"
echo "========================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies if needed
pip install numpy scipy --quiet

# Run temporal augmentation
echo "Applying temporal augmentation to StatsBomb data..."
python scripts/augment_statsbomb_temporal.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Phase 2.4 Complete!"
    echo "Output: data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl"
else
    echo ""
    echo "✗ Phase 2.4 Failed!"
    exit 1
fi

echo ""
echo "End time: $(date)"
