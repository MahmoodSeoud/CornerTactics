#!/bin/bash
# Setup script for soccersegcal pipeline
# This creates a fresh virtual environment with all dependencies

set -e

PIPELINE_DIR="/home/mseo/CornerTactics/soccersegcal-pipeline"
VENV_DIR="${PIPELINE_DIR}/venv"

echo "=== Setting up soccersegcal pipeline environment ==="

# Load required modules
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

# Create virtual environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3.9 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "Installing base packages..."
pip install --upgrade pip wheel setuptools

# Core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib tqdm
pip install opencv-python-headless
pip install scikit-learn

# SoccerNet
pip install SoccerNet

# YOLOv8 for player detection
pip install ultralytics

# Clone and install soccersegcal
cd "${PIPELINE_DIR}"
if [ ! -d "soccersegcal" ]; then
    echo "Cloning soccersegcal..."
    git clone https://github.com/Spiideo/soccersegcal.git
fi

cd soccersegcal
pip install -r requirements.txt 2>/dev/null || true
pip install -e . 2>/dev/null || pip install . 2>/dev/null || echo "Note: soccersegcal installed with warnings"

# Clone and install sskit
cd "${PIPELINE_DIR}"
if [ ! -d "sskit" ]; then
    echo "Cloning sskit..."
    git clone https://github.com/Spiideo/sskit.git
fi

cd sskit
pip install -e . 2>/dev/null || pip install . 2>/dev/null || echo "Note: sskit installed with warnings"

# Download pretrained calibration model
cd "${PIPELINE_DIR}"
mkdir -p models
if [ ! -f "models/soccersegcal_snapshot.ckpt" ]; then
    echo "Downloading pretrained calibration model..."
    wget -O models/soccersegcal_snapshot.ckpt \
        https://github.com/Spiideo/soccersegcal/releases/download/SoccerNetChallenge2023/snapshot.ckpt
fi

echo ""
echo "=== Setup complete ==="
echo "To activate: source ${VENV_DIR}/bin/activate"
echo "Corner clips available: $(ls /home/mseo/CornerTactics/data/corner_clips/*.mp4 | wc -l)"
