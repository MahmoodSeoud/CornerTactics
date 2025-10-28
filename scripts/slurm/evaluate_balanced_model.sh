#!/bin/bash
#SBATCH --job-name=eval_balanced
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_balanced_%j.out
#SBATCH --error=logs/eval_balanced_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Evaluating Balanced GNN Model"
echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install torch torch-geometric scikit-learn tqdm --quiet

# Run evaluation
python scripts/evaluate_balanced_model.py

echo ""
echo "=============================="
echo "End time: $(date)"