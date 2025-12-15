#!/bin/bash
#SBATCH --job-name=soccersegcal_test
#SBATCH --output=/home/mseo/CornerTactics/logs/slurm/soccersegcal_test_%j.out
#SBATCH --error=/home/mseo/CornerTactics/logs/slurm/soccersegcal_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Quick test of soccersegcal pipeline on 10 corners
# Use this to verify everything works before full run

set -e

echo "=== SoccerSegCal Quick Test ==="
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

# Create directories
mkdir -p /home/mseo/CornerTactics/logs/slurm
mkdir -p "${PIPELINE_DIR}/data"
mkdir -p "${PIPELINE_DIR}/outputs"

# Test with limit of 10 corners
LIMIT=10

echo ""
echo "=== Testing with ${LIMIT} corners ==="

echo ""
echo "Step 1: Load corners..."
python scripts/01_load_corners.py

echo ""
echo "Step 2: Extract frames (limit ${LIMIT})..."
python scripts/02_extract_frames.py --limit ${LIMIT}

echo ""
echo "Step 3: Camera calibration..."
python scripts/03_calibrate_cameras.py --device cuda --limit $((LIMIT * 3))

echo ""
echo "Step 4: Player detection..."
python scripts/04_detect_players.py --model yolov8m.pt --visualize --limit $((LIMIT * 3))

echo ""
echo "Step 5: Project to pitch..."
python scripts/05_project_to_pitch.py --min-players 5 --visualize

echo ""
echo "Step 6: Create dataset..."
python scripts/06_create_dataset.py

echo ""
echo "=== Quick Test Complete ==="
echo "End time: $(date)"

# Show results
echo ""
echo "=== Results ==="
ls -la "${PIPELINE_DIR}/data/"

if [ -f "${PIPELINE_DIR}/data/corner_dataset.json" ]; then
    echo ""
    echo "Dataset created successfully!"
    python -c "import json; d=json.load(open('${PIPELINE_DIR}/data/corner_dataset.json')); print(f'Dataset samples: {len(d)}')"
else
    echo "Warning: Dataset not created"
fi
