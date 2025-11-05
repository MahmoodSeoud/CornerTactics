#!/bin/bash
#SBATCH --job-name=viz_feat_imp
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/viz_feat_imp_%j.out
#SBATCH --error=logs/viz_feat_imp_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "Visualizing XGBoost Feature Importance"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project root
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install matplotlib mplsoccer numpy --quiet

# Run visualization
echo "Running feature importance visualization..."
python scripts/visualization/visualize_feature_importance.py \
    --importance-path data/results/feature_importance.json \
    --output-dir data/results/feature_importance

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
