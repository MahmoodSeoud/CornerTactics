#!/bin/bash
#SBATCH --job-name=ussf_bl
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/ussf_bl_rerun_%j.out
#SBATCH --error=logs/ussf_bl_rerun_%j.err

# =============================================================================
# Re-run USSF-aligned baselines (MLP + XGBoost) with corrected layout detection
# 5 seeds, combined dataset
# =============================================================================

echo "=============================================="
echo "USSF Baselines Re-run (corrected code)"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

FM="--feature-mode ussf_aligned --combined"

for SEED in 42 123 456 789 1234; do
    echo ""
    echo "--- Baselines seed=$SEED ---"
    python3 -u -m corner_prediction.run_all --baselines all --seed $SEED $FM
done

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
