#!/bin/bash
#SBATCH --job-name=viz_graphs
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/visualize_graphs_%j.out
#SBATCH --error=logs/visualize_graphs_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "=================================================="
echo "Graph Visualization Job"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Load GCC module for GLIBCXX compatibility
module load GCCcore/11.3.0 2>/dev/null || echo "GCC module not available, continuing..."

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Fix library path for pandas compatibility
export LD_LIBRARY_PATH=/opt/itu/easybuild/software/Anaconda3/2024.02-1/lib:$LD_LIBRARY_PATH

echo ""
echo "Environment activated: robo"
echo "Working directory: $(pwd)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Install dependencies quietly
echo "Installing dependencies..."
pip install matplotlib pandas numpy scipy mplsoccer --quiet

echo ""
echo "=================================================="
echo "Running graph visualization..."
echo "=================================================="
echo ""

# Visualize all adjacency strategies for 5 sample corners
python scripts/visualization/visualize_graph_structure.py --strategy all --num-samples 5

echo ""
echo "=================================================="
echo "Visualization Complete!"
echo "End time: $(date)"
echo "=================================================="
echo ""
echo "Output files saved to: data/results/graphs/"
echo "Check for files named: strategy_comparison_*.png"
