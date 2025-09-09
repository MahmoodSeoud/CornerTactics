#!/bin/bash
#SBATCH --job-name=soccernet_720p
#SBATCH --output=logs/download_720p_%j.log
#SBATCH --error=logs/download_720p_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Load any required modules
module load python/3.9

# Set work directory
cd /home/mseo/CornerTactics

# Download with your password
PASSWORD="s0cc3rn3t"

echo "Starting 720p download at $(date)"
echo "Node: $(hostname)"

# Test network speed first
echo "Testing network speed..."
curl -s https://www.google.com -o /dev/null -w "Speed: %{speed_download} bytes/s\n"

# Download 720p videos (larger files - will take much longer)
echo "Downloading 720p broadcast videos..."
python src/download_soccernet.py --videos 720p --password "$PASSWORD"

echo "720p download completed at $(date)"