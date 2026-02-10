#!/bin/bash
#SBATCH --job-name=dfl_download
#SBATCH --output=logs/dfl_download_%j.out
#SBATCH --error=logs/dfl_download_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=scavenge

# Download DFL tracking data from figshare
# Data is ~2.5GB total across 21 files

cd /home/mseo/CornerTactics

echo "Starting DFL data download at $(date)"
echo "Hostname: $(hostname)"

python scripts/download_dfl_data.py

echo "Download completed at $(date)"
