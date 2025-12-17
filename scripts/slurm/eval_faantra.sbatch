#!/bin/bash
#SBATCH --job-name=faantra_eval
#SBATCH --partition=acltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/faantra_eval_%j.out
#SBATCH --error=logs/faantra_eval_%j.err

# ============================================================================
# Evaluate FAANTRA on corner kick outcome prediction
#
# Usage:
#   CHECKPOINT=checkpoints/corners/corner_baseline/best.ckpt sbatch scripts/slurm/eval_faantra.sbatch
#
#   # Evaluate on different split
#   CHECKPOINT=... SPLIT=val sbatch scripts/slurm/eval_faantra.sbatch
# ============================================================================

echo "=== FAANTRA Evaluation ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
date
nvidia-smi

# Configuration
CONFIG=${CONFIG:-config/SoccerNetBall/Corner-Config-v2.json}
MODEL=${MODEL:-corner_baseline}
CHECKPOINT=${CHECKPOINT:?Error: CHECKPOINT must be set}
SPLIT=${SPLIT:-test}

cd /home/mseo/CornerTactics/FAANTRA
source venv/bin/activate

# Load CUDA
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"

python test.py "$CONFIG" "$CHECKPOINT" "$MODEL" -s "$SPLIT"

echo "=== Evaluation Complete ==="
date
