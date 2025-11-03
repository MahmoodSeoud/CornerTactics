#!/bin/bash
#SBATCH --job-name=relabel_recv_v2
#SBATCH --partition=scavenge
#SBATCH --output=logs/relabel_receivers_v2_%j.out
#SBATCH --error=logs/relabel_receivers_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Re-label Receivers Using Event Streams (v2)
# Increases coverage from 60% to 85%+ by including defensive clearances

echo "=========================================================================="
echo "RE-LABELING RECEIVERS USING EVENT STREAMS (v2)"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================================================="

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install statsbombpy pandas numpy tqdm --quiet

# Run re-labeling
echo ""
echo "Running event-stream-based receiver labeling..."
python scripts/preprocessing/add_receiver_labels_v2.py

echo ""
echo "=========================================================================="
echo "Job completed at: $(date)"
echo "=========================================================================="
