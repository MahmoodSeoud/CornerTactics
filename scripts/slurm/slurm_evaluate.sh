#!/bin/bash
#SBATCH --job-name=evaluate_models_49
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

# Load required modules
module load GCC/14.2.0

# Activate conda environment
source ~/.bashrc
conda activate robo

# Fix library path for pandas compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run evaluation
echo "Starting model evaluation..."
echo "Start time: $(date)"

python scripts/06_evaluate_models.py

echo "Evaluation complete!"
echo "End time: $(date)"

# List generated results
echo "Generated results:"
ls -lh results/*.png results/*.json results/*.md 2>/dev/null || echo "No results found"
