#!/bin/bash
#SBATCH --job-name=ablation_exps
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/ablation_exps_%j.out
#SBATCH --error=logs/ablation_exps_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load GCC module for library compatibility
module --ignore_cache load GCC/14.2.0 || module load GCCcore/14.2.0 || echo "Warning: GCC module not loaded"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install xgboost scikit-learn pandas numpy tqdm --quiet

# Create output directory
mkdir -p results/ablation_experiments

# Run ablation experiments (all 10 configs)
echo "Starting ablation experiments..."
echo "Date: $(date)"
echo "="

python scripts/run_ablation_experiments.py

echo ""
echo "Ablation experiments complete!"
echo "Date: $(date)"
