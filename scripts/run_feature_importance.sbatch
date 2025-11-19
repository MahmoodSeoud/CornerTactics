#!/bin/bash
#SBATCH --job-name=feature-importance
#SBATCH --output=logs/feature_importance_%j.out
#SBATCH --error=logs/feature_importance_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scavenge

# Feature Importance Analysis for Corner Kick Shot Prediction
# Runs comprehensive feature importance with multiple methods

echo "========================================"
echo "Feature Importance Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd /home/mseo/CornerTactics

# Create directories
mkdir -p logs
mkdir -p results/feature_importance/figures

# Load required modules
module purge
module load GCC/14.2.0
module load Anaconda3

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robo

# Print environment info
echo ""
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Install required packages if not already installed
pip install xgboost shap --quiet 2>/dev/null || true

# Run the analysis
python scripts/feature_importance_analysis.py \
    --data data/processed/corners_features_with_shot.csv \
    --output results/feature_importance

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Analysis completed successfully!"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "Output files:"
    echo "  - results/feature_importance/feature_importance_report.md"
    echo "  - results/feature_importance/feature_analysis_results.json"
    echo "  - results/feature_importance/xgboost_clean_baseline.pkl"
    echo "  - results/feature_importance/figures/*.png"
else
    echo ""
    echo "========================================"
    echo "Analysis FAILED with exit code $?"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi