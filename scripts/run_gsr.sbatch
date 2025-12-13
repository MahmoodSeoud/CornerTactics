#!/bin/bash
#SBATCH --job-name=gsr_corners
#SBATCH --output=logs/slurm/gsr_%A_%a.out
#SBATCH --error=logs/slurm/gsr_%A_%a.err
#SBATCH --array=0-83  # 4229 clips / 50 per job = 85 jobs (0-84), rounded to 0-83 for 4200
#SBATCH --time=05:00:00  # 50 clips × 6 min = 300 min = 5 hours
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr
#SBATCH --account=students
#SBATCH --cpus-per-task=4

# GSR Batch Processing Script
# Processes corner clips through the sn-gamestate pipeline

# Load required modules
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

# Configuration
VIDEOS_PER_JOB=50
CLIPS_DIR="/home/mseo/CornerTactics/data/corner_clips"
OUTPUT_DIR="/home/mseo/CornerTactics/outputs"
VENV_DIR="/home/mseo/CornerTactics/sn-gamestate/.venv"

# Calculate start/end indices for this array task
START_IDX=$((SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB))
END_IDX=$((START_IDX + VIDEOS_PER_JOB - 1))

echo "============================================"
echo "GSR Batch Processing - Array Task ${SLURM_ARRAY_TASK_ID}"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Processing clips ${START_IDX} to ${END_IDX}"
echo "Start time: $(date)"
echo "============================================"

# Create output directories
mkdir -p ${OUTPUT_DIR}/states
mkdir -p ${OUTPUT_DIR}/logs

# Activate virtual environment
source ${VENV_DIR}/bin/activate

# Get list of clips to process
cd /home/mseo/CornerTactics

# Find all corner clips and sort them
CLIPS=($(ls -1 ${CLIPS_DIR}/corner_*.mp4 2>/dev/null | sort))
TOTAL_CLIPS=${#CLIPS[@]}

echo "Total clips available: ${TOTAL_CLIPS}"

# Process clips in this chunk
PROCESSED=0
SKIPPED=0
FAILED=0

for ((i=START_IDX; i<=END_IDX && i<TOTAL_CLIPS; i++)); do
    CLIP_PATH="${CLIPS[$i]}"
    CLIP_NAME=$(basename "${CLIP_PATH}" .mp4)
    CORNER_ID=$(echo "${CLIP_NAME}" | sed 's/corner_/CORNER-/')
    STATE_FILE="${OUTPUT_DIR}/states/${CORNER_ID}.pklz"

    # Skip if already processed
    if [ -f "${STATE_FILE}" ]; then
        echo "Skipping ${CORNER_ID} (already processed)"
        ((SKIPPED++))
        continue
    fi

    echo "[$(date +%H:%M:%S)] Processing ${CORNER_ID} (${i}/${TOTAL_CLIPS})"

    # Run tracklab inference
    cd /home/mseo/CornerTactics/sn-gamestate

    tracklab -cn soccernet \
        dataset=video \
        dataset.video_path="${CLIP_PATH}" \
        dataset.eval_set=val \
        eval_tracking=false \
        state.save_file="${STATE_FILE}" \
        2>&1 | tee "${OUTPUT_DIR}/logs/${CORNER_ID}.log"

    if [ -f "${STATE_FILE}" ]; then
        echo "  -> Success: $(ls -lh ${STATE_FILE} | awk '{print $5}')"
        ((PROCESSED++))
    else
        echo "  -> Failed: No state file generated"
        ((FAILED++))
    fi
done

echo ""
echo "============================================"
echo "Array task ${SLURM_ARRAY_TASK_ID} complete!"
echo "  Processed: ${PROCESSED}"
echo "  Skipped: ${SKIPPED}"
echo "  Failed: ${FAILED}"
echo "End time: $(date)"
echo "============================================"
