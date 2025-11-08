#!/bin/bash
#SBATCH --job-name=xgb_ablation_all
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/xgb_ablation_all_%j.out
#SBATCH --error=logs/xgb_ablation_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12

# XGBoost Ablation Studies - All Three Experiments
# 1. Temporal Augmentation Impact (t=0s vs 5 frames)
# 2. Feature Selection Impact (22 players vs 5 closest)
# 3. Feature Importance Analysis (which features matter)

echo "================================================================"
echo "XGBoost Ablation Studies - All Experiments"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 24GB"
echo "Start time: $(date)"
echo "================================================================"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install xgboost scikit-learn matplotlib --quiet

# Run all experiments
echo ""
echo "================================================================"
echo "Running all three ablation experiments..."
echo "================================================================"
python scripts/analysis/xgboost_ablation_experiments.py \
    --graph-path data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl \
    --output-dir results/ablation_studies \
    --experiments all \
    --random-seed 42

echo ""
echo "================================================================"
echo "Experiments complete! Results saved to results/ablation_studies/"
echo "================================================================"
echo "End time: $(date)"
echo "================================================================"
