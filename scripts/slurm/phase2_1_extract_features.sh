#!/bin/bash
#SBATCH --job-name=phase2_1_features
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/phase2_1_features_%j.out
#SBATCH --error=logs/phase2_1_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Phase 2.1: Node Feature Engineering
# Extracts 14-dimensional feature vectors per player from corner kick datasets

echo "=========================================="
echo "Phase 2.1: Node Feature Engineering"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install required dependencies
echo "Installing dependencies..."
pip install pandas numpy tqdm pyarrow --quiet

# Run feature extraction
echo ""
echo "Starting feature extraction..."
echo ""
python scripts/extract_corner_features.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature extraction completed successfully"
    echo "Output files created in data/features/node_features/"
else
    echo ""
    echo "❌ Feature extraction failed"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "=========================================="
