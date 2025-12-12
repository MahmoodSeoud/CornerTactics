#!/bin/bash
#SBATCH --job-name=soccernet_download
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err
#SBATCH --time=23:59:00
#SBATCH --partition=scavenge
#SBATCH --account=students
#SBATCH --requeue
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

# SoccerNet Video Download Job
# Downloads all 550 games (720p) from SoccerNet
# Expected: ~500GB total, 1100 video files

# IMPORTANT: Set your NDA password before running:
#   export SOCCERNET_PASSWORD="your_password"
#   sbatch scripts/download_soccernet.sbatch
#
# Or edit this file to set it directly (not recommended for security):
#   SOCCERNET_PASSWORD="your_password"

# Configuration
SCRIPT_DIR="/home/mseo/CornerTactics/scripts"
DATA_DIR="/home/mseo/CornerTactics/data/misc/soccernet/videos"
LOG_DIR="/home/mseo/CornerTactics/logs/slurm"
VENV_DIR="/home/mseo/CornerTactics/sn-gamestate/.venv"

# Create log directory
mkdir -p ${LOG_DIR}

# Print job info
echo "============================================"
echo "SoccerNet Video Download Job"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Data directory: ${DATA_DIR}"
echo "============================================"

# Set password directly
SOCCERNET_PASSWORD="s0cc3rn3t"
export SOCCERNET_PASSWORD

# Activate virtual environment
echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# Check current download status
echo ""
echo "Checking current download status..."
python ${SCRIPT_DIR}/download_soccernet_videos.py --check-only --local-dir ${DATA_DIR}

# Start download
echo ""
echo "Starting download..."
python ${SCRIPT_DIR}/download_soccernet_videos.py \
    --local-dir ${DATA_DIR} \
    --password "${SOCCERNET_PASSWORD}" \
    --splits train valid test challenge \
    --max-retries 20

# Check final status
echo ""
echo "============================================"
echo "Download complete!"
echo "End time: $(date)"
echo "============================================"
python ${SCRIPT_DIR}/download_soccernet_videos.py --check-only --local-dir ${DATA_DIR}

# Count final video files
echo ""
echo "Final video count:"
find ${DATA_DIR} -name "*_720p.mkv" | wc -l
