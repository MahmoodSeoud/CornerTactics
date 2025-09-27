#!/bin/bash
#SBATCH --job-name=gsr_setup
#SBATCH --output=logs/gsr_setup_%j.out
#SBATCH --error=logs/gsr_setup_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dgpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Install dependencies for SoccerNet Game State Reconstruction
# This only needs to be run once or when dependencies change

mkdir -p logs
cd /home/mseo/CornerTactics/sn-gamestate

echo "=== Installing GSR Dependencies ==="
echo "Installing base dependencies..."
uv pip install -e .

echo "Installing MMCV for computer vision models..."
uv run mim install mmcv==2.0.1

echo "Installing SoccerNet package for dataset downloads..."
uv pip install SoccerNet

echo "=== Dependency Installation Complete ==="
echo "Installed packages:"
uv pip list | grep -E "(torch|transformers|tracklab|mmcv|SoccerNet)"

echo "Python version:"
uv run python --version

echo "PyTorch GPU availability:"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Dependencies successfully installed!"