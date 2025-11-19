#!/bin/bash
#SBATCH --job-name=feature-importance-gpu
#SBATCH --output=logs/feature_importance_gpu_%j.out
#SBATCH --error=logs/feature_importance_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scavenge

# Feature Importance Analysis with Fresh Environment
# Creates new conda env and runs comprehensive analysis

echo "========================================"
echo "Feature Importance Analysis (Fresh Environment)"
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
module load Anaconda3

# Activate conda base
source $(conda info --base)/etc/profile.d/conda.sh

# Check if environment exists, create if not
ENV_NAME="cornertactics"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} exists, activating..."
    conda activate ${ENV_NAME}
else
    echo "Creating new environment ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}

    echo "Installing packages..."
    # Install core packages
    pip install --upgrade pip
    pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0
    pip install matplotlib==3.7.2 seaborn==0.12.2
    pip install scipy==1.11.1 joblib

    # Install ML packages
    pip install xgboost==1.7.6
    pip install shap==0.42.1

    echo "Packages installed successfully"
fi

# Print environment info
echo ""
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Verify imports work
echo "Verifying package imports..."
python -c "
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import shap
print('NumPy version:', np.__version__)
print('Pandas version:', pd.__version__)
print('Scikit-learn version:', sklearn.__version__)
print('XGBoost version:', xgb.__version__)
print('SHAP version:', shap.__version__)
print('All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "Package import failed! Exiting..."
    exit 1
fi

echo ""
echo "========================================"
echo "Running Feature Importance Analysis"
echo "========================================"
echo ""

# Run the comprehensive analysis
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
    echo "  - results/feature_importance/figures/"
    echo ""
    echo "Key visualizations:"
    echo "  - feature_importance_comparison.png"
    echo "  - shap_summary.png"
    echo "  - shap_importance_bar.png"

    # Also run the simple version for comparison
    echo ""
    echo "Running simple RandomForest version for comparison..."
    python scripts/feature_importance_simple.py

else
    echo ""
    echo "========================================"
    echo "Analysis FAILED with exit code $?"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "All analyses complete!"
echo "========================================"