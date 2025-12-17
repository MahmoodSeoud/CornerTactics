#!/bin/bash
#SBATCH --job-name=frames
#SBATCH --partition=scavenge
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/frames_%A_%a.out
#SBATCH --error=logs/frames_%A_%a.err

# ============================================================================
# Extract frames from video clips for FAANTRA training
#
# This array job extracts frames in parallel chunks.
# Default: 10 chunks for train split (~387 clips each)
#
# Usage:
#   # Extract train split (default)
#   sbatch scripts/slurm/extract_frames.sbatch
#
#   # Extract specific split
#   SPLIT=valid CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch
#
# Monitor:
#   squeue -u $USER
#   watch -n 30 'ls FAANTRA/data/corner_anticipation/train/clip_*/ | wc -l'
# ============================================================================

echo "=== Frame Extraction: Chunk $SLURM_ARRAY_TASK_ID ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
date

# Configuration (can be overridden via environment)
SPLIT=${SPLIT:-train}
CHUNKS=${CHUNKS:-10}

# Setup environment
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

echo "Split: $SPLIT, Chunk: $SLURM_ARRAY_TASK_ID/$CHUNKS"

# Run extraction
python scripts/03_prepare_faantra_data.py \
    --extract-frames \
    --split $SPLIT \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks $CHUNKS \
    --workers 8

echo "=== Done ==="
date
