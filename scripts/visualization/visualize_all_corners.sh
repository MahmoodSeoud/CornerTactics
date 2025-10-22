#!/bin/bash
#SBATCH --job-name=viz_all_corners
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/viz_all_corners_%j.out
#SBATCH --error=logs/viz_all_corners_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install mplsoccer matplotlib pandas tqdm --quiet

# Run batch visualization
python scripts/visualize_all_corners.py
