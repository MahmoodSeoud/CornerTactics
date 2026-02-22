#!/bin/bash
#SBATCH --job-name=ussf_abl
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/ussf_abl_ms_%A_%a.out
#SBATCH --error=logs/ussf_abl_ms_%A_%a.err

# =============================================================================
# USSF-Aligned Multi-Seed Ablations + Baseline Permutation Tests
#
# Submit all in parallel:
#   sbatch --array=0-2 scripts/slurm/run_ussf_ablation_multiseed.sbatch
#
# Array tasks:
#   0 = Ablation: ussf_position_only (5 seeds) — WITH edge masking fix
#   1 = Ablation: ussf_plus_velocity (5 seeds) — full features baseline
#   2 = Baseline permutation tests (MLP + XGBoost, USSF-aligned, combined)
# =============================================================================

echo "=============================================="
echo "USSF Ablation Multi-Seed: Task $SLURM_ARRAY_TASK_ID"
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
        echo ">>> Ablation: ussf_position_only (5 seeds, with edge masking)"
        for SEED in 42 123 456 789 1234; do
            echo ""
            echo "--- ussf_position_only seed=$SEED ---"
            python3 -u -m corner_prediction.run_all --ablation ussf_position_only --seed $SEED $FM
        done
        ;;
    1)
        echo ">>> Ablation: ussf_plus_velocity (5 seeds)"
        for SEED in 42 123 456 789 1234; do
            echo ""
            echo "--- ussf_plus_velocity seed=$SEED ---"
            python3 -u -m corner_prediction.run_all --ablation ussf_plus_velocity --seed $SEED $FM
        done
        ;;
    2)
        echo ">>> Baseline Permutation Tests (USSF-aligned, combined)"
        python3 -u -m corner_prediction.run_all --baseline-permutation both --seed 42 $FM
        ;;
    *)
        echo "ERROR: Unknown task ID $SLURM_ARRAY_TASK_ID (expected 0-2)"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
