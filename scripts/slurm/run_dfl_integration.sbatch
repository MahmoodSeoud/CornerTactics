#!/bin/bash
#SBATCH --job-name=dfl_integ
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dfl_integ_%A_%a.out
#SBATCH --error=logs/dfl_integ_%A_%a.err

# =============================================================================
# Task 7: DFL Integration — Extract, Merge, and Evaluate
#
# Submit all tasks sequentially (extraction must complete before evaluation):
#   sbatch --array=0 scripts/slurm/run_dfl_integration.sbatch    # Extract + Merge
#   # Wait for task 0 to complete, then:
#   sbatch --array=1-2 scripts/slurm/run_dfl_integration.sbatch  # Eval + Permutation
#
# Or submit extraction + evaluation in one shot:
#   sbatch --array=0 scripts/slurm/run_dfl_integration.sbatch
#   sbatch --dependency=afterok:$JOBID --array=1-2 scripts/slurm/run_dfl_integration.sbatch
#
# Array tasks:
#   0 = DFL extraction + merge (CPU-only, no GPU needed)
#   1 = LOMO evaluation on combined dataset (pretrained backbone)
#   2 = Permutation tests on combined dataset (n=100)
# =============================================================================

echo "=============================================="
echo "DFL Integration (Task 7): Task $SLURM_ARRAY_TASK_ID"
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

COMMON_ARGS="--seed 42 --combined"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo ">>> Step 1: DFL Extraction + Merge"
        echo "Extracting DFL corners from raw tracking data and merging with SkillCorner..."
        python3 -u -m corner_prediction.run_all --extract-dfl --seed 42
        ;;
    1)
        echo ">>> Step 2: LOMO Evaluation (combined dataset, pretrained backbone)"
        python3 -u -m corner_prediction.run_all --eval-only --mode pretrained $COMMON_ARGS
        ;;
    2)
        echo ">>> Step 3: Permutation Tests (combined dataset)"
        python3 -u -m corner_prediction.run_all --permutation-only --permutation-target both \
            --n-permutations 100 --mode pretrained $COMMON_ARGS
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
