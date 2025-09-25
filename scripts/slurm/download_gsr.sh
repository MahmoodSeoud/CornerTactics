#!/bin/bash
#SBATCH --job-name=gsr_download
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/gsr_download_%j.out
#SBATCH --error=logs/gsr_download_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

cd /home/mseo/CornerTactics/scripts
python gsr_download_gamestate-2024.py
