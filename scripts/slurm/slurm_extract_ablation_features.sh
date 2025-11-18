#!/bin/bash
#SBATCH --job-name=extract_ablation
#SBATCH --output=/home/mseo/CornerTactics/logs/extract_ablation_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/extract_ablation_%j.err
#SBATCH --partition=scavenge
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Navigate to project directory
cd /home/mseo/CornerTactics

# Run feature extraction
echo "Starting progressive feature extraction..."
python scripts/extract_features_progressive.py

echo "Feature extraction complete!"
