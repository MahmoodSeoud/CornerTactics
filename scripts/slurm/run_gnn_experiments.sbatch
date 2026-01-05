#!/bin/bash
#SBATCH --job-name=gnn_baseline
#SBATCH --output=logs/gnn_baseline_%j.out
#SBATCH --error=logs/gnn_baseline_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# GNN Baseline Experiments for Corner Kick Outcome Prediction
# Tests GraphSAGE and GAT with different graph structures

echo "=============================================="
echo "GNN Baseline Experiments"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="

# Setup environment
cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Run experiments
echo "Running GNN experiments..."
python3 experiments/gnn_baseline/run_experiments.py \
    --hidden 64 \
    --epochs 100 \
    --patience 15 \
    --seed 42 \
    --verbose

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
