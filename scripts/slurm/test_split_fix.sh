#!/bin/bash
#SBATCH --job-name=test_split_fix
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/test_split_fix_%j.out
#SBATCH --error=logs/test_split_fix_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Testing data split fix for data leakage..."
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
pip install torch torch-geometric scikit-learn --quiet

# Run test script
echo ""
echo "Running data leakage verification test..."
echo "=========================================="
python scripts/test_split_fix.py

exit_code=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"

exit $exit_code
