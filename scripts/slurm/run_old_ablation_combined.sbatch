#!/bin/bash
#SBATCH --job-name=old_abl
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/old_ablation_%A_%a.out
#SBATCH --error=logs/old_ablation_%A_%a.err

# =============================================================================
# Old 13-feature schema velocity ablation on combined dataset (5 seeds each)
#
# Submit: sbatch --array=0-9 scripts/slurm/run_old_ablation_combined.sbatch
#
# Array tasks:
#   0-4 = position_only (seeds 42, 123, 456, 789, 1234)
#   5-9 = plus_velocity (seeds 42, 123, 456, 789, 1234)
# =============================================================================

SEEDS=(42 123 456 789 1234)

if [ $SLURM_ARRAY_TASK_ID -lt 5 ]; then
    CONFIG="position_only"
    SEED_IDX=$SLURM_ARRAY_TASK_ID
else
    CONFIG="plus_velocity"
    SEED_IDX=$((SLURM_ARRAY_TASK_ID - 5))
fi

SEED=${SEEDS[$SEED_IDX]}

echo "=============================================="
echo "Old 13-feat ablation: $CONFIG, seed=$SEED"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

python3 -u -m corner_prediction.run_all \
    --ablation "$CONFIG" --combined --seed "$SEED"

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
