#!/bin/bash
#SBATCH --job-name=add_receiver_labels
#SBATCH --partition=acltr
#SBATCH --output=logs/add_receiver_labels_%j.out
#SBATCH --error=logs/add_receiver_labels_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set working directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Run receiver label extraction
echo "Starting receiver label extraction..."
python scripts/preprocessing/add_receiver_labels.py

echo "Receiver label extraction complete!"
