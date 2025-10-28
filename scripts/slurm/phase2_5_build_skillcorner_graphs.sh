#!/bin/bash
#SBATCH --job-name=phase2_5_sc_graphs
#SBATCH --partition=acltr
#SBATCH --account=researchers
#SBATCH --output=logs/phase2_5_sc_graphs_%j.out
#SBATCH --error=logs/phase2_5_sc_graphs_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Phase 2.5: Build SkillCorner Graphs and Merge with StatsBomb
# Converts SkillCorner temporal features into graph representations
# Merges with StatsBomb augmented graphs to create combined dataset

echo "========================================"
echo "Phase 2.5: Build SkillCorner Graphs"
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
pip install numpy scipy pandas --quiet

# Build SkillCorner graphs and merge datasets
echo "Building SkillCorner graphs from temporal features..."
python scripts/build_skillcorner_graphs.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Phase 2.5 Complete!"
    echo "Output: data/graphs/adjacency_team/combined_temporal_graphs.pkl"
    echo "Dataset: 7,369 graphs (StatsBomb + SkillCorner)"
else
    echo ""
    echo "✗ Phase 2.5 Failed!"
    exit 1
fi

echo ""
echo "End time: $(date)"
