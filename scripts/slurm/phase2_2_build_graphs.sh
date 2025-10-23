#!/bin/bash
#SBATCH --job-name=phase2_2_graphs
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/phase2_2_graphs_%j.out
#SBATCH --error=logs/phase2_2_graphs_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Phase 2.2: Build Graph Dataset with Adjacency Matrices
# Converts node features to graph representations for GNN training

echo "============================================================"
echo "Phase 2.2: Graph Dataset Construction"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install scipy tqdm --quiet

# Create logs directory if it doesn't exist
mkdir -p logs

# Default adjacency strategy (can override with: sbatch --export=STRATEGY=distance ...)
STRATEGY=${STRATEGY:-team}

echo ""
echo "Adjacency Strategy: $STRATEGY"
echo ""

# Build graphs with selected strategy
echo "Building graph dataset..."
python scripts/build_graph_dataset.py \
    --strategy $STRATEGY \
    --dataset all \
    --features-dir data/features/node_features \
    --output-dir data/graphs/adjacency_${STRATEGY}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Phase 2.2 Complete - Graph dataset built successfully"
    echo "============================================================"
    echo "Output directory: data/graphs/adjacency_${STRATEGY}/"
    echo ""
    echo "Files created:"
    ls -lh data/graphs/adjacency_${STRATEGY}/
    echo ""
    echo "Next: Phase 3 - GNN Model Implementation"
    echo "  1. Set up Spektral environment"
    echo "  2. Implement GNN architecture (src/gnn_model.py)"
    echo "  3. Create training pipeline (scripts/train_gnn.py)"
else
    echo ""
    echo "============================================================"
    echo "❌ Phase 2.2 Failed - Check error log"
    echo "============================================================"
    echo "Error log: logs/phase2_2_graphs_${SLURM_JOB_ID}.err"
    exit 1
fi
