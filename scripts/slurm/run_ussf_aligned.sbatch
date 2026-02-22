#!/bin/bash
#SBATCH --job-name=ussf_align
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/ussf_aligned_%A_%a.out
#SBATCH --error=logs/ussf_aligned_%A_%a.err

# =============================================================================
# USSF-Aligned Feature Engineering Experiments (Experiment A)
#
# Submit all in parallel:
#   sbatch --array=0-4 scripts/slurm/run_ussf_aligned.sbatch
#
# Submit individually:
#   sbatch --array=0 scripts/slurm/run_ussf_aligned.sbatch  # Multi-seed eval
#   sbatch --array=1 scripts/slurm/run_ussf_aligned.sbatch  # Ablation: position_only
#   sbatch --array=2 scripts/slurm/run_ussf_aligned.sbatch  # Ablation: plus_velocity
#   sbatch --array=3 scripts/slurm/run_ussf_aligned.sbatch  # Permutation test
#   sbatch --array=4 scripts/slurm/run_ussf_aligned.sbatch  # Baselines (5 seeds)
#
# Array tasks:
#   0 = Multi-seed LOMO evaluation (5 seeds, combined)
#   1 = Ablation: ussf_position_only (combined)
#   2 = Ablation: ussf_plus_velocity (combined)
#   3 = Permutation test (combined, n=100)
#   4 = Baselines: MLP + XGBoost (5 seeds each, combined)
# =============================================================================

echo "=============================================="
echo "USSF-Aligned Experiments: Task $SLURM_ARRAY_TASK_ID"
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

FM="--feature-mode ussf_aligned --combined"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> Multi-seed LOMO (ussf_aligned, combined, 5 seeds)"
        python3 -u -m corner_prediction.run_all --multi-seed $FM
        ;;
    1)
        echo ">>> Ablation: ussf_position_only"
        python3 -u -m corner_prediction.run_all --ablation ussf_position_only --seed 42 $FM
        ;;
    2)
        echo ">>> Ablation: ussf_plus_velocity"
        python3 -u -m corner_prediction.run_all --ablation ussf_plus_velocity --seed 42 $FM
        ;;
    3)
        echo ">>> Permutation Test (ussf_aligned, combined, n=100)"
        python3 -u -m corner_prediction.run_all --permutation-only --n-permutations 100 --seed 42 $FM
        ;;
    4)
        echo ">>> Baselines (ussf_aligned, combined, 5 seeds)"
        for SEED in 42 123 456 789 1234; do
            echo ""
            echo "--- Baselines seed=$SEED ---"
            python3 -u -m corner_prediction.run_all --baselines all --seed $SEED $FM
        done
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
