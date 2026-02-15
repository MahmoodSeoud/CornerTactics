#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=short
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --error=logs/baselines_%A_%a.err

# =============================================================================
# Baseline Comparisons for Corner Kick Prediction
#
# Submit individual baselines:
#   sbatch --array=0 scripts/slurm/run_baselines.sbatch    # Random
#   sbatch --array=1 scripts/slurm/run_baselines.sbatch    # Heuristic
#   sbatch --array=2 scripts/slurm/run_baselines.sbatch    # XGBoost
#   sbatch --array=3 --partition=dgpu --gres=gpu:1 scripts/slurm/run_baselines.sbatch  # MLP (GPU)
#
# Submit CPU baselines together:
#   sbatch --array=0-2 scripts/slurm/run_baselines.sbatch
#
# Submit all (sequential):
#   sbatch --array=4 scripts/slurm/run_baselines.sbatch
#
# Array tasks:
#   0 = Random baseline (no training, CPU)
#   1 = Heuristic receiver (no training, CPU)
#   2 = XGBoost aggregate (sklearn, CPU)
#   3 = MLP flat features (PyTorch, GPU recommended)
#   4 = All baselines sequentially
# =============================================================================

echo "=============================================="
echo "Baselines: Task $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/corner_prediction

# Report GPU info if available
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

COMMON_ARGS="--seed 42"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> Random Baseline"
        python3 -u -m corner_prediction.baselines.run_baselines --baseline random $COMMON_ARGS
        ;;
    1)
        echo ">>> Heuristic Receiver Baseline"
        python3 -u -m corner_prediction.baselines.run_baselines --baseline heuristic $COMMON_ARGS
        ;;
    2)
        echo ">>> XGBoost Baseline"
        python3 -u -m corner_prediction.baselines.run_baselines --baseline xgboost $COMMON_ARGS
        ;;
    3)
        echo ">>> MLP Baseline"
        python3 -u -m corner_prediction.baselines.run_baselines --baseline mlp $COMMON_ARGS
        ;;
    4)
        echo ">>> All Baselines (sequential)"
        python3 -u -m corner_prediction.baselines.run_baselines --baseline all $COMMON_ARGS
        ;;
    *)
        echo "ERROR: Unknown task ID $SLURM_ARRAY_TASK_ID (expected 0-4)"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
