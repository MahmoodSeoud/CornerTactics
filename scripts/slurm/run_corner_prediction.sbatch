#!/bin/bash
#SBATCH --job-name=corner_pred
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/corner_pred_%A_%a.out
#SBATCH --error=logs/corner_pred_%A_%a.err

# =============================================================================
# Two-Stage Corner Kick Prediction: Training & Evaluation
#
# Submit as array job:
#   sbatch --array=0-7 scripts/slurm/run_corner_prediction.sbatch
#
# Or submit individual tasks:
#   sbatch --array=0 scripts/slurm/run_corner_prediction.sbatch    # LOMO pretrained
#   sbatch --array=1 scripts/slurm/run_corner_prediction.sbatch    # LOMO scratch
#   sbatch --array=2-5 scripts/slurm/run_corner_prediction.sbatch  # Ablations
#   sbatch --array=6 scripts/slurm/run_corner_prediction.sbatch    # Permutation (receiver)
#   sbatch --array=7 scripts/slurm/run_corner_prediction.sbatch    # Permutation (shot)
#
# Array tasks:
#   0 = LOMO evaluation (pretrained backbone)
#   1 = LOMO evaluation (scratch backbone)
#   2 = Ablation: position_only
#   3 = Ablation: plus_velocity
#   4 = Ablation: full_fc_edges
#   5 = All ablations (sequential)
#   6 = Permutation test: receiver (n=100)
#   7 = Permutation test: shot (n=100)
# =============================================================================

echo "=============================================="
echo "Corner Kick Prediction: Task $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/corner_prediction

# Report GPU info
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

COMMON_ARGS="--seed 42"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> LOMO Evaluation (pretrained backbone)"
        python3 -u -m corner_prediction.run_all --eval-only --mode pretrained $COMMON_ARGS
        ;;
    1)
        echo ">>> LOMO Evaluation (scratch backbone)"
        python3 -u -m corner_prediction.run_all --eval-only --mode scratch $COMMON_ARGS
        ;;
    2)
        echo ">>> Ablation: position_only"
        python3 -u -m corner_prediction.run_all --ablation position_only --mode pretrained $COMMON_ARGS
        ;;
    3)
        echo ">>> Ablation: plus_velocity"
        python3 -u -m corner_prediction.run_all --ablation plus_velocity --mode pretrained $COMMON_ARGS
        ;;
    4)
        echo ">>> Ablation: full_fc_edges"
        python3 -u -m corner_prediction.run_all --ablation full_fc_edges --mode pretrained $COMMON_ARGS
        ;;
    5)
        echo ">>> All Ablations (sequential)"
        python3 -u -m corner_prediction.run_all --all-ablations --mode pretrained $COMMON_ARGS
        ;;
    6)
        echo ">>> Permutation Test: Receiver (n=100)"
        python3 -u -m corner_prediction.run_all --permutation-only --permutation-target receiver \
            --n-permutations 100 --mode pretrained $COMMON_ARGS
        ;;
    7)
        echo ">>> Permutation Test: Shot (n=100)"
        python3 -u -m corner_prediction.run_all --permutation-only --permutation-target shot \
            --n-permutations 100 --mode pretrained $COMMON_ARGS
        ;;
    *)
        echo "ERROR: Unknown task ID $SLURM_ARRAY_TASK_ID (expected 0-7)"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
