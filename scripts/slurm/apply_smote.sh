#!/bin/bash
#SBATCH --job-name=apply_smote
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/apply_smote_%j.out
#SBATCH --error=logs/apply_smote_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Applying SMOTE to corner kick graphs..."
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install imbalanced-learn scikit-learn torch --quiet

# Run SMOTE
python scripts/apply_smote.py

if [ $? -eq 0 ]; then
    echo ""
    echo "SMOTE application complete!"
    echo "Output saved to: data/graphs/adjacency_team/combined_temporal_graphs_smote.pkl"
else
    echo "ERROR: SMOTE application failed!"
    exit 1
fi

echo "End time: $(date)"