#!/bin/bash
#SBATCH --job-name=feature-importance-fixed
#SBATCH --output=logs/feature_importance_fixed_%j.out
#SBATCH --error=logs/feature_importance_fixed_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=scavenge

# Feature Importance Analysis with OpenBLAS Fix

echo "========================================"
echo "Feature Importance Analysis (OpenBLAS Fix)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd /home/mseo/CornerTactics

# Create directories
mkdir -p logs
mkdir -p results/feature_importance/figures

# Load required modules INCLUDING OpenBLAS
module purge
module load GCC/13.2.0  # Use a GCC version that exists
module load OpenBLAS/0.3.24-GCC-13.2.0  # Matching OpenBLAS
module load Anaconda3

# Set library paths for OpenBLAS
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH

# Find and export OpenBLAS library path explicitly
OPENBLAS_LIB=$(find $EBROOTOPENBLAS -name "libopenblas*.so" 2>/dev/null | head -1)
if [ -n "$OPENBLAS_LIB" ]; then
    echo "Found OpenBLAS library: $OPENBLAS_LIB"
    export LD_LIBRARY_PATH=$(dirname $OPENBLAS_LIB):$LD_LIBRARY_PATH
else
    echo "Warning: Could not find OpenBLAS library"
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robo

# Print environment info
echo ""
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Test numpy import
echo "Testing NumPy import..."
python -c "import numpy as np; print('NumPy version:', np.__version__); print('NumPy works!')"

if [ $? -ne 0 ]; then
    echo "NumPy import failed. Trying to fix with conda..."

    # Try reinstalling numpy from conda-forge
    conda install -y -c conda-forge numpy=1.24.3 --force-reinstall

    # Test again
    python -c "import numpy as np; print('NumPy version:', np.__version__)"

    if [ $? -ne 0 ]; then
        echo "Still failing. Trying pip reinstall..."
        pip uninstall -y numpy
        pip install numpy==1.24.3
    fi
fi

# Install XGBoost and SHAP if needed
python -c "import xgboost" 2>/dev/null || pip install xgboost==1.7.6
python -c "import shap" 2>/dev/null || pip install shap

echo ""
echo "========================================"
echo "Running Feature Importance Analysis"
echo "========================================"
echo ""

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

    # Also run simple version
    echo ""
    echo "Running simple RandomForest version..."
    python scripts/feature_importance_simple.py
else
    echo ""
    echo "========================================"
    echo "Feature importance analysis failed"
    echo "Trying simple RandomForest version as fallback..."
    echo "========================================"
    python scripts/feature_importance_simple.py
fi