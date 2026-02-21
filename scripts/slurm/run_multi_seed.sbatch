#!/bin/bash
#SBATCH --job-name=multi-seed
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/multi_seed_%j.out
#SBATCH --error=logs/multi_seed_%j.err

# =============================================================================
# Multi-Seed LOMO Evaluation (5 seeds, combined dataset)
#
# Submit:
#   sbatch scripts/slurm/run_multi_seed.sbatch
# =============================================================================

echo "=============================================="
echo "Multi-Seed LOMO Evaluation"
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

echo ">>> Multi-seed LOMO (combined, pretrained, 5 seeds)"
python3 -u -m corner_prediction.run_all --multi-seed --combined --mode pretrained

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
