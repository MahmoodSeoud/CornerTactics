#!/bin/bash
#SBATCH --job-name=gsr_corner
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gsr_%A_%a.out
#SBATCH --error=logs/gsr_%A_%a.err

# ============================================================================
# Run sn-gamestate TrackLab pipeline on a batch of corner kick clips.
#
# Usage:
#   # Pilot (5 batches of 10 clips)
#   sbatch --array=0-4 scripts/slurm/run_gsr_batch.sbatch
#
#   # Full run (85 batches of 50 clips)
#   sbatch --array=0-84 scripts/slurm/run_gsr_batch.sbatch
#
#   # Use scavenge partition for opportunistic scheduling
#   sbatch --array=0-84 --partition=scavenge --time=04:00:00 scripts/slurm/run_gsr_batch.sbatch
#
# Prerequisites:
#   - Run prepare_gsr_batches.py first to create batch directories
#   - sn-gamestate uv venv must be set up (sn-gamestate/.venv)
#
# Monitor:
#   squeue -u $USER
#   ls sn-gamestate/outputs/corner_states/*.pklz | wc -l
# ============================================================================

echo "=== GSR Corner Batch: $SLURM_ARRAY_TASK_ID ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Configuration
PROJECT_DIR=/home/mseo/CornerTactics
BATCH_DIR=${PROJECT_DIR}/sn-gamestate/data/corner_batches
BATCH_ID=$(printf "batch_%04d" $SLURM_ARRAY_TASK_ID)
STATE_DIR=${PROJECT_DIR}/sn-gamestate/outputs/corner_states

# Check batch exists
if [ ! -d "${BATCH_DIR}/${BATCH_ID}" ]; then
    echo "ERROR: Batch directory not found: ${BATCH_DIR}/${BATCH_ID}"
    exit 1
fi

CLIP_COUNT=$(ls "${BATCH_DIR}/${BATCH_ID}"/*.mp4 2>/dev/null | wc -l)
echo "Processing batch: ${BATCH_ID} (${CLIP_COUNT} clips)"

# Skip if already completed
if [ -f "${STATE_DIR}/${BATCH_ID}.pklz" ]; then
    echo "State file already exists: ${STATE_DIR}/${BATCH_ID}.pklz — skipping"
    exit 0
fi

# Setup environment
mkdir -p "${STATE_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Use sn-gamestate uv venv (has all deps including hydra, tracklab, sn-gamestate)
cd ${PROJECT_DIR}/sn-gamestate
source .venv/bin/activate

# Run TrackLab
# - dataset=video uses ExternalVideo (supports directory of .mp4 files)
# - Override video_path to point to this batch
# - Save state as pklz for later extraction
# - Disable visualization to save time
# - Process all videos in batch (nvid=-1)
# - Disable evaluation (no ground truth)
tracklab -cn soccernet \
    dataset=video \
    "dataset.video_path=${BATCH_DIR}/${BATCH_ID}" \
    "state.save_file=${STATE_DIR}/${BATCH_ID}.pklz" \
    "visualization.cfg.save_videos=False" \
    "dataset.nvid=-1" \
    "eval_tracking=False" \
    "test_tracking=False"

EXIT_CODE=$?
echo "=== Done (exit code: $EXIT_CODE) ==="
date
