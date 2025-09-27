#!/bin/bash
#SBATCH --job-name=gsr_baseline
#SBATCH --partition=acltr
#SBATCH --account=researchers
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --output=logs/gsr_baseline_%j.out
#SBATCH --error=logs/gsr_baseline_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

mkdir -p logs
cd /home/mseo/CornerTactics/sn-gamestate

# Fix SSL certificates for UV container environment
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

echo "=== Running SoccerNet GSR Baseline ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run GSR baseline - will auto-download dataset & models on first run
uv run tracklab -cn soccernet

echo "GSR baseline complete!"
echo "Results location: outputs/"