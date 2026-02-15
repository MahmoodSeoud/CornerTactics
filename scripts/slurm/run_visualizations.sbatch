#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --partition=short
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err

# =============================================================================
# Generate Thesis-Ready Figures from Corner Kick Prediction Results
#
# Submit:
#   sbatch scripts/slurm/run_visualizations.sbatch
#
# Reads result pickles from results/corner_prediction/ and generates
# PDF figures to results/corner_prediction/figures/.
#
# CPU-only job — no GPU required.
# =============================================================================

echo "=============================================="
echo "Visualization: Figure Generation"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/corner_prediction/figures

python3 -c "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
echo ""

python3 -u -m corner_prediction.visualization.generate_all

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
