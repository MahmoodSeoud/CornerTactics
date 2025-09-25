#!/bin/bash
#SBATCH --job-name=gsr_single_corner
#SBATCH --output=logs/gsr_corner_test_%j.log
#SBATCH --error=logs/gsr_corner_test_%j.err
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to sn-gamestate directory
cd /home/mseo/CornerTactics/sn-gamestate

# Fix SSL certificates for UV container
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Auto-answer 'no' to download prompt since datasets are already present
# Use tvcalib instead of nbjw_calib to avoid keypoint passing issues
echo "n" | uv run tracklab -cn soccernet 

