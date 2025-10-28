#!/bin/bash
#SBATCH --job-name=test_split_integrity
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/test_split_integrity_%j.out
#SBATCH --error=logs/test_split_integrity_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set project root
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "======================================================================"
echo "Testing Train/Val/Test Split Integrity"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Run integrity test
python scripts/test_receiver_split_integrity.py

echo ""
echo "Completed at: $(date)"
echo "======================================================================"
