#!/bin/bash
#SBATCH --job-name=shot_prm
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/shot_perm_%A_%a.out
#SBATCH --error=logs/shot_perm_%A_%a.err

# =============================================================================
# Shot-Only Permutation Tests (receiver already completed)
#
# Submit both:
#   sbatch --array=0-1 scripts/slurm/run_shot_perms.sbatch
#
# Array tasks:
#   0 = USSF-aligned shot permutation (combined, n=100)
#   1 = Adapter-based shot permutation (combined, n=100)
# =============================================================================

echo "=============================================="
echo "Shot Permutation Test: Task $SLURM_ARRAY_TASK_ID"
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

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> Shot Permutation Test (USSF-aligned, combined, n=100)"
        python3 -u -m corner_prediction.run_all \
            --permutation-only --permutation-target shot \
            --n-permutations 100 --seed 42 \
            --feature-mode ussf_aligned --combined
        ;;
    1)
        echo ">>> Shot Permutation Test (adapter/pretrained, combined, n=100)"
        python3 -u -m corner_prediction.run_all \
            --permutation-only --permutation-target shot \
            --n-permutations 100 --seed 42 \
            --combined
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
