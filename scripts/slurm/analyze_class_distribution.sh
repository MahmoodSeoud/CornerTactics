#!/bin/bash
#SBATCH --job-name=class_dist
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/class_dist_%j.out
#SBATCH --error=logs/class_dist_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Analyzing class distribution in corner kick dataset..."
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Change to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install torch torch-geometric scikit-learn --quiet

# Run analysis script
echo ""
echo "Running class distribution analysis..."
echo "======================================================"
python scripts/analyze_class_distribution.py

echo ""
echo "======================================================"
echo "End time: $(date)"
