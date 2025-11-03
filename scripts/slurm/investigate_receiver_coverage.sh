#!/bin/bash
#SBATCH --job-name=receiver_coverage
#SBATCH --partition=scavenge
#SBATCH --output=logs/receiver_coverage_%j.out
#SBATCH --error=logs/receiver_coverage_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Investigate Receiver Label Coverage
# Analyzes why only 60% of graphs have receiver labels

echo "=================================================="
echo "Investigating Receiver Label Coverage"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install pandas numpy tqdm scikit-learn --quiet

# Run analysis
echo ""
echo "Running receiver coverage analysis..."
python scripts/analysis/investigate_receiver_coverage.py

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="
