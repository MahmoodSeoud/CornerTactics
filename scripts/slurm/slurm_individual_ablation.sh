#!/bin/bash
#SBATCH --job-name=individual_ablation
#SBATCH --output=/home/mseo/CornerTactics/logs/individual_ablation_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/individual_ablation_%j.err
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Navigate to project directory
cd /home/mseo/CornerTactics

# Run individual feature ablation
echo "Starting individual feature ablation analysis..."
echo "This will test:"
echo "  - Phase 1: 27 raw features (leave-one-out)"
echo "  - Phase 2: 34 engineered features (univariate)"
echo "  - Phase 3: Forward selection for minimal set"
echo ""
echo "Total models to train: ~122 (61 features × 2 tasks, RF only)"
echo ""

python scripts/11_individual_feature_ablation.py

echo ""
echo "Individual feature ablation complete!"
