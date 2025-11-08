#!/bin/bash
#SBATCH --job-name=xgb_ablation_exp1
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/xgb_ablation_exp1_%j.out
#SBATCH --error=logs/xgb_ablation_exp1_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# XGBoost Ablation Study - Experiment 1: Temporal Augmentation Impact
# Compares t=0s only vs 5 temporal frames for outcome prediction

echo "================================================================"
echo "XGBoost Ablation - Experiment 1: Temporal Augmentation"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Run experiment
echo "Running Experiment 1..."
python scripts/analysis/xgboost_ablation_experiments.py \
    --graph-path data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl \
    --output-dir results/ablation_studies \
    --experiments 1 \
    --random-seed 42

echo "================================================================"
echo "End time: $(date)"
echo "================================================================"
