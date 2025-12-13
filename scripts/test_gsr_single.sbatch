#!/bin/bash
#SBATCH --job-name=gsr_test
#SBATCH --output=logs/slurm/gsr_test_%j.out
#SBATCH --error=logs/slurm/gsr_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=acltr
#SBATCH --account=students
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Test GSR pipeline on a single corner clip

# Load required modules for mmcv compatibility
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

DATASET_PATH="/home/mseo/CornerTactics/data/MySoccerNetGS"
OUTPUT_DIR="/home/mseo/CornerTactics/outputs"
VENV_DIR="/home/mseo/CornerTactics/sn-gamestate/.venv"

echo "============================================"
echo "GSR Single Video Test"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "============================================"

# Create output directories
mkdir -p ${OUTPUT_DIR}/states
mkdir -p ${OUTPUT_DIR}/json

# Activate virtual environment
source ${VENV_DIR}/bin/activate

# Run GSR on CORNER-0000
cd /home/mseo/CornerTactics/sn-gamestate

# Check huggingface-hub version
echo ""
echo "huggingface-hub version:"
pip show huggingface-hub | grep Version

echo ""
echo "Running GSR on CORNER-0000..."
# Use ExternalVideo wrapper for direct video file processing
VIDEO_PATH="/home/mseo/CornerTactics/data/corner_clips/corner_0000.mp4"

tracklab -cn soccernet \
    dataset=video \
    dataset.video_path=${VIDEO_PATH} \
    dataset.eval_set=val \
    eval_tracking=false \
    state.save_file=${OUTPUT_DIR}/states/CORNER-0000.pklz

echo ""
echo "============================================"
echo "Test complete!"
echo "End time: $(date)"
echo "============================================"

# Check outputs
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/states/CORNER-0000.pklz 2>/dev/null || echo "No state file found"
