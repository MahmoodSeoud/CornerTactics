#!/bin/bash
#SBATCH --job-name=soccersegcal
#SBATCH --output=/home/mseo/CornerTactics/logs/slurm/soccersegcal_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/slurm/soccersegcal_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# SoccerSegCal Pipeline - Full Run
# Processes all corner clips through the calibration and detection pipeline

set -e

echo "=== SoccerSegCal Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup environment
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

PIPELINE_DIR="/home/mseo/CornerTactics/soccersegcal-pipeline"
cd "$PIPELINE_DIR"

# Activate virtual environment
source "${PIPELINE_DIR}/venv/bin/activate"

# Create log directory
mkdir -p /home/mseo/CornerTactics/logs/slurm

echo ""
echo "=== Step 1: Load Corners ==="
python scripts/01_load_corners.py

echo ""
echo "=== Step 2: Extract Frames ==="
python scripts/02_extract_frames.py

echo ""
echo "=== Step 3: Camera Calibration ==="
python scripts/03_calibrate_cameras.py --device cuda

echo ""
echo "=== Step 4: Player Detection ==="
python scripts/04_detect_players.py --model yolov8x.pt

echo ""
echo "=== Step 5: Project to Pitch ==="
python scripts/05_project_to_pitch.py --visualize

echo ""
echo "=== Step 6: Create Dataset ==="
python scripts/06_create_dataset.py

echo ""
echo "=== Pipeline Complete ==="
echo "End time: $(date)"
echo "Output directory: ${PIPELINE_DIR}/data/"
