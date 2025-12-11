#!/bin/bash
#SBATCH --job-name=gsr_corners
#SBATCH --output=logs/slurm/gsr_%A_%a.out
#SBATCH --error=logs/slurm/gsr_%A_%a.err
#SBATCH --array=0-99  # Will be adjusted based on total videos
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Configuration
VIDEOS_PER_JOB=50  # Process 50 videos per array task
DATASET_PATH="/home/mseo/CornerTactics/data/MySoccerNetGS"
OUTPUT_DIR="/home/mseo/CornerTactics/outputs"
VIDEO_LIST="${DATASET_PATH}/video_list.txt"

# Calculate start/end indices for this array task
START_IDX=$((SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB))
END_IDX=$((START_IDX + VIDEOS_PER_JOB - 1))

# Load conda environment
source ~/.bashrc
conda activate sn-gamestate

# Create output directories
mkdir -p ${OUTPUT_DIR}/states
mkdir -p ${OUTPUT_DIR}/json

# Get total number of videos
TOTAL_VIDEOS=$(wc -l < ${VIDEO_LIST})

# Process videos in this chunk
for ((i=START_IDX; i<=END_IDX && i<TOTAL_VIDEOS; i++)); do
    VIDEO_ID=$(sed -n "$((i+1))p" ${VIDEO_LIST})

    if [ -z "${VIDEO_ID}" ]; then
        continue
    fi

    # Skip if already processed
    if [ -f "${OUTPUT_DIR}/json/${VIDEO_ID}.json" ]; then
        echo "Skipping ${VIDEO_ID} (already processed)"
        continue
    fi

    echo "Processing ${VIDEO_ID} (${i}/${TOTAL_VIDEOS})"

    # Run tracklab inference
    cd /home/mseo/CornerTactics/sn-gamestate
    uv run tracklab -cn soccernet \
        dataset_path=${DATASET_PATH} \
        eval_set=custom \
        nvid=${i} \
        evaluator=null \
        save_videos=false \
        state.save_file=${OUTPUT_DIR}/states/${VIDEO_ID}.pklz \
        2>&1 | tee ${OUTPUT_DIR}/logs/${VIDEO_ID}.log

    # The JSON output should be in the default location
    # Copy it to our organized output directory if it exists
done

echo "Array task ${SLURM_ARRAY_TASK_ID} complete"
