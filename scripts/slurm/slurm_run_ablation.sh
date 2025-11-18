#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --output=/home/mseo/CornerTactics/logs/ablation_study_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/ablation_study_%j.err
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_40gb:1

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Navigate to project directory
cd /home/mseo/CornerTactics

# Run ablation study
echo "Starting ablation study (10 steps × 2 tasks × 3 models = 60 training runs)..."
python scripts/09_ablation_study.py

echo "Ablation study complete!"
