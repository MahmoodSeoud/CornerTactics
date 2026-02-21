#!/bin/bash
#SBATCH --job-name=cbl-perm
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/combined_bl_perm_%A_%a.out
#SBATCH --error=logs/combined_bl_perm_%A_%a.err

# =============================================================================
# Permutation Tests for Baseline Models on COMBINED Dataset (143 corners)
#
# Array tasks:
#   0 = MLP       (GPU, ~8h with 100 perms on combined)
#   1 = XGBoost   (CPU, ~1h with 100 perms on combined)
#
# Submit MLP (needs GPU):
#   sbatch --array=0 scripts/slurm/run_combined_baseline_permutation.sbatch
#
# Submit XGBoost (CPU-only):
#   sbatch --array=1 --gres=gpu:0 scripts/slurm/run_combined_baseline_permutation.sbatch
#
# Submit both:
#   sbatch --array=0-1 scripts/slurm/run_combined_baseline_permutation.sbatch
# =============================================================================

echo "=============================================="
echo "Combined Baseline Permutation Test: Task $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/corner_prediction

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

N_PERMS=${N_PERMS:-100}
SEED=${SEED:-42}
COMMON_ARGS="--combined --n-permutations $N_PERMS --seed $SEED"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> MLP Permutation Test on COMBINED (n=$N_PERMS)"
        python3 -u -m corner_prediction.run_all --baseline-permutation mlp $COMMON_ARGS
        ;;
    1)
        echo ">>> XGBoost Permutation Test on COMBINED (n=$N_PERMS)"
        python3 -u -m corner_prediction.run_all --baseline-permutation xgboost --no-gpu $COMMON_ARGS
        ;;
    *)
        echo "ERROR: Unknown task ID $SLURM_ARRAY_TASK_ID (expected 0-1)"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
