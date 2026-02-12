#!/bin/bash
#SBATCH --job-name=ussf_train
#SBATCH --partition=dgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --gres=gpu:rtx2070:1
#SBATCH --time=06:00:00
#SBATCH --output=transfer_learning/logs/phase1_%j.out
#SBATCH --error=transfer_learning/logs/phase1_%j.err
#SBATCH --account=researchers

# ============================================================================
# Phase 1: Train USSF Backbone
#
# Train CrystalConv GNN on USSF counterattack data for transfer learning.
# Trains both dense and normal adjacency types by default.
#
# Usage:
#   sbatch transfer_learning/slurm/phase1_train_backbone.sbatch
#
#   # Train only specific adjacency type:
#   ADJ_TYPE=dense sbatch transfer_learning/slurm/phase1_train_backbone.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f transfer_learning/logs/phase1_*.out
# ============================================================================

echo "=== Phase 1: Train USSF Backbone ==="
echo "Job: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Configuration (can be overridden via environment)
ADJ_TYPE=${ADJ_TYPE:-all}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-3}

# Setup environment
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Create logs directory
mkdir -p transfer_learning/logs

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run training
if [ "$ADJ_TYPE" = "all" ]; then
    echo "Training BOTH dense and normal adjacency types"
    python -u transfer_learning/phase1_train_ussf_backbone.py \
        --train-all \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR
else
    echo "Training $ADJ_TYPE adjacency type only"
    python -u transfer_learning/phase1_train_ussf_backbone.py \
        --adj-type $ADJ_TYPE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR
fi

echo ""
echo "=== Done ==="
date
