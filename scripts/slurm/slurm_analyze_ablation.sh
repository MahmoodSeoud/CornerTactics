#!/bin/bash
#SBATCH --job-name=analyze_ablation
#SBATCH --output=/home/mseo/CornerTactics/logs/analyze_ablation_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/analyze_ablation_%j.err
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Navigate to project directory
cd /home/mseo/CornerTactics

# Run analysis
echo "Starting ablation analysis..."
python scripts/10_analyze_ablation.py

echo "Analysis complete!"
