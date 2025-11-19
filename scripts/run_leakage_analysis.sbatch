#!/bin/bash
#SBATCH --job-name=leakage-analysis
#SBATCH --output=logs/leakage_analysis_%j.out
#SBATCH --error=logs/leakage_analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Data Leakage Analysis for Corner Kick Shot Prediction
# This script runs comprehensive leakage detection on the feature set

echo "========================================"
echo "Data Leakage Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd /home/mseo/CornerTactics

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p reports/figures

# Load required modules
module purge
module load Anaconda3

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robo

# Print environment info
echo ""
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Run the analysis
# Option 1: With actual data file
# python scripts/analyze_data_leakage.py --data data/processed/corner_features.csv --output reports

# Option 2: Generate theoretical report (no data file required)
python scripts/analyze_data_leakage.py --output reports

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Analysis completed successfully!"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "Output files:"
    echo "  - reports/data_leakage_report.md"
    echo "  - reports/leakage_analysis_results.json"
    echo "  - reports/figures/leakage_heatmap.png"
    echo "  - reports/figures/mcc_vs_importance.png"
    echo "  - reports/figures/timeline_diagram.png"
else
    echo ""
    echo "========================================"
    echo "Analysis FAILED with exit code $?"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi
