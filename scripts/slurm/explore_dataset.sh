#!/bin/bash
#SBATCH --job-name=explore_dataset
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/explore_dataset_%j.out
#SBATCH --error=logs/explore_dataset_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load GCC module for library compatibility
module load GCC/14.2.0

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install pandas numpy scipy --quiet

# Create output directory
mkdir -p results/exploration

# Run exploration script
echo "Starting dataset exploration..."
echo "Date: $(date)"
echo "="

python scripts/explore_dataset.py

echo ""
echo "Exploration complete!"
echo "Date: $(date)"
