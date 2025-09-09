#!/bin/bash
#SBATCH --job-name=soccernet_download
#SBATCH --partition=dgpu
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load any required modules
module load python/3.9

# Activate virtual environment if you have one
# source venv/bin/activate

# Set work directory
cd /home/mseo/CornerTactics

# Download with your password
PASSWORD="s0cc3rn3t"

echo "Starting download at $(date)"
echo "Node: $(hostname)"

# Test network speed first
echo "Testing network speed..."
curl -s https://www.google.com -o /dev/null -w "Speed: %{speed_download} bytes/s\n"

# Download 224p videos (smaller, faster)
python src/download_soccernet.py --videos 224p --password "$PASSWORD" --data-dir data_224p

# Download tracklets
python src/download_soccernet.py --tracklets tracking --data-dir data_224p

echo "Download completed at $(date)"
