#!/bin/bash
#SBATCH --job-name=gsr_download
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/gsr_download_%j.out
#SBATCH --error=logs/gsr_download_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Download and prepare SoccerNet Game State dataset
# Handles SSL certificates and dataset version mismatch

mkdir -p logs
cd /home/mseo/CornerTactics/sn-gamestate

# Fix SSL certificates for UV container environment
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

echo "=== Downloading SoccerNet Game State Dataset ==="
echo "SSL certificate: $SSL_CERT_FILE"

# Check if we need to download or just finish unzipping
if [ ! -d data/SoccerNetGS/valid ]; then
    echo "Starting fresh download of SoccerNet Game State dataset..."

    uv run python -c "
from SoccerNet.Downloader import SoccerNetDownloader
import os

print('Creating downloader...')
downloader = SoccerNetDownloader(LocalDirectory='data/SoccerNetGS')

print('Downloading gamestate-2024 validation set (11.2GB)...')
try:
    downloader.downloadDataTask(task='gamestate-2024', split=['valid'])
    print('Download completed successfully!')
except Exception as e:
    print(f'Download error: {e}')
    exit(1)
"

    if [ $? -ne 0 ]; then
        echo "Download failed!"
        exit 1
    fi

    # Create version symlink and unzip
    cd data/SoccerNetGS
    echo "Creating gamestate-2025 symlink..."
    ln -s gamestate-2024 gamestate-2025

    echo "Unzipping validation data (this may take 30-45 minutes)..."
    unzip -q gamestate-2024/valid.zip -d valid
    cd ../..

elif [ -f data/SoccerNetGS/gamestate-2024/valid.zip ]; then
    echo "Found existing download, continuing unzipping..."
    cd data/SoccerNetGS

    # Check if symlink exists
    if [ ! -L gamestate-2025 ]; then
        echo "Creating gamestate-2025 symlink..."
        ln -s gamestate-2024 gamestate-2025
    fi

    # Continue unzipping if it was interrupted
    echo "Resuming unzipping process..."
    unzip -o -q gamestate-2024/valid.zip -d valid
    cd ../..
    echo "Unzipping completed!"

else
    echo "Dataset already exists in data/SoccerNetGS/valid"
fi

# Verify dataset integrity
echo ""
echo "=== Dataset Verification ==="
if [ -d data/SoccerNetGS/valid ]; then
    echo "Validation sequences found:"
    ls data/SoccerNetGS/valid | grep SNGS | wc -l

    echo "Dataset size:"
    du -sh data/SoccerNetGS/valid

    echo "Sample sequence structure:"
    ls -la data/SoccerNetGS/valid/SNGS-023 2>/dev/null | head -5 || echo "SNGS-023 not found"
else
    echo "ERROR: Dataset directory not found!"
    exit 1
fi

echo ""
echo "Dataset preparation complete!"
echo "Ready to run: sbatch run_gsr_pipeline.sh"