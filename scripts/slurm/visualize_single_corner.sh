#!/bin/bash
#SBATCH --job-name=viz_single_corner
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/viz_single_corner_%j.out
#SBATCH --error=logs/viz_single_corner_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install mplsoccer matplotlib pandas --quiet

# Run single corner visualization
python scripts/visualize_single_corner.py
