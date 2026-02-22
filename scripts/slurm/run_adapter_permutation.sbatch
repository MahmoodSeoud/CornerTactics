#!/bin/bash
#SBATCH --job-name=adpt_perm
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/adapter_perm_%j.out
#SBATCH --error=logs/adapter_perm_%j.err

# =============================================================================
# Adapter-Based (Non-USSF-Aligned) Permutation Test on Combined Dataset
#
# Submit:
#   sbatch scripts/slurm/run_adapter_permutation.sbatch
# =============================================================================

echo "=============================================="
echo "Adapter-Based Permutation Test (combined)"
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

echo ">>> Permutation Test (pretrained/adapter, combined, n=100)"
python3 -u -m corner_prediction.run_all --permutation-only --n-permutations 100 --seed 42 --combined

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
