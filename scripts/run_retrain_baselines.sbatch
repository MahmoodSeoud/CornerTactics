#!/bin/bash
#SBATCH --job-name=retrain-baselines
#SBATCH --output=logs/retrain_baselines_%j.out
#SBATCH --error=logs/retrain_baselines_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scavenge

# Retrain all baseline models without leaked features

echo "========================================"
echo "Retraining Clean Baseline Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd /home/mseo/CornerTactics

# Create output directories
mkdir -p logs
mkdir -p models/clean_baselines/figures

# Load required modules
module purge
module load GCC/13.2.0
module load OpenBLAS/0.3.24-GCC-13.2.0
module load Anaconda3

# Set library paths for OpenBLAS
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robo

# Print environment info
echo ""
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Install XGBoost if needed
python -c "import xgboost" 2>/dev/null || pip install xgboost==1.7.6

# Run the retraining script
echo "Starting model retraining..."
echo "========================================"
python scripts/retrain_clean_baselines.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Retraining completed successfully!"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "Output files:"
    echo "  - models/clean_baselines/mlp_clean.pkl"
    echo "  - models/clean_baselines/random_forest_clean.pkl"
    echo "  - models/clean_baselines/xgboost_clean.pkl"
    echo "  - models/clean_baselines/clean_baseline_results.json"
    echo "  - models/clean_baselines/figures/"
else
    echo ""
    echo "========================================"
    echo "Retraining FAILED with exit code $?"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi