#!/bin/bash
#SBATCH --job-name=faantra
#SBATCH --partition=acltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/faantra_%j.out
#SBATCH --error=logs/faantra_%j.err

# ============================================================================
# Train FAANTRA on corner kick outcome prediction
#
# Usage:
#   sbatch scripts/slurm/train_faantra.sbatch
#
#   # With custom config and model name
#   CONFIG=config/SoccerNetBall/Corner-Config-v2.json MODEL=corner_v2 sbatch scripts/slurm/train_faantra.sbatch
#
#   # Resume from checkpoint
#   CHECKPOINT=checkpoints/corners/corner_model/checkpoint_epoch_10.ckpt sbatch scripts/slurm/train_faantra.sbatch
# ============================================================================

echo "=== FAANTRA Training ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
date
nvidia-smi

# Configuration (can be overridden via environment)
CONFIG=${CONFIG:-config/SoccerNetBall/Corner-Config-v2.json}
MODEL=${MODEL:-corner_baseline}
CHECKPOINT=${CHECKPOINT:-}

cd /home/mseo/CornerTactics/FAANTRA
source venv/bin/activate

# Load CUDA
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1

echo "Config: $CONFIG"
echo "Model: $MODEL"
echo "Checkpoint: ${CHECKPOINT:-None}"

# Run training
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint..."
    python main.py "$CONFIG" "$MODEL" --checkpoint-path "$CHECKPOINT"
else
    echo "Starting fresh training..."
    python main.py "$CONFIG" "$MODEL"
fi

echo "=== Training Complete ==="
date
