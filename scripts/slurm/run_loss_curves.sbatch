#!/bin/bash
#SBATCH --job-name=loss_curves
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/loss_curves_%j.out
#SBATCH --error=logs/loss_curves_%j.err

# =============================================================================
# Re-run GNN & MLP training to capture loss curves, then regenerate figures.
#
# Submit:
#   sbatch scripts/slurm/run_loss_curves.sbatch
#
# Runs sequentially:
#   1. GNN LOMO (SkillCorner-only)     → captures loss history
#   2. GNN LOMO (combined)             → captures loss history
#   3. MLP baseline (SkillCorner-only) → captures loss history
#   4. MLP baseline (combined)         → captures loss history
#   5. Generate all figures            → includes new loss curve plots
# =============================================================================

echo "=============================================="
echo "Loss Curve Capture + Figure Regeneration"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/corner_prediction/figures

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

# --- 1. GNN LOMO (SkillCorner-only) ---
echo "=========================================="
echo ">>> [1/5] GNN LOMO (SkillCorner-only)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --eval-only --mode pretrained --seed 42

# --- 2. GNN LOMO (combined) ---
echo ""
echo "=========================================="
echo ">>> [2/5] GNN LOMO (combined)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --eval-only --mode pretrained --seed 42 --combined

# --- 3. MLP baseline (SkillCorner-only) ---
echo ""
echo "=========================================="
echo ">>> [3/5] MLP baseline (SkillCorner-only)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --baselines mlp --seed 42

# --- 4. MLP baseline (combined) ---
echo ""
echo "=========================================="
echo ">>> [4/5] MLP baseline (combined)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --baselines mlp --seed 42 --combined

# --- 5. Generate all figures ---
echo ""
echo "=========================================="
echo ">>> [5/5] Generate all figures"
echo "=========================================="
python3 -u -m corner_prediction.visualization.generate_all

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
