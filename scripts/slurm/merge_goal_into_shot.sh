#!/bin/bash
#SBATCH --job-name=merge_goals
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/merge_goals_%j.out
#SBATCH --error=logs/merge_goals_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "Merging Goal labels into Shot category..."
echo "=========================================="
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
pip install pandas scipy --quiet

# Run merge script (standalone version)
python scripts/merge_goal_into_shot_standalone.py

echo ""
echo "=========================================="
echo "End time: $(date)"
