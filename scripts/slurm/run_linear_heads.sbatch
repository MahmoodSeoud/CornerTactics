#!/bin/bash
#SBATCH --job-name=linear_heads
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/linear_heads_%j.out
#SBATCH --error=logs/linear_heads_%j.err

# =============================================================================
# Linear Heads Evaluation: Address overfitting by replacing MLP heads with
# single linear layers (linear probes).
#
# Submit:
#   sbatch scripts/slurm/run_linear_heads.sbatch
#
# Runs sequentially:
#   1. GNN LOMO with linear heads (combined)         → main comparison
#   2. GNN LOMO with linear heads (SkillCorner-only)  → ablation baseline
#   3. MLP baseline with linear probe (combined)      → baseline comparison
#   4. MLP baseline with linear probe (SkillCorner)   → baseline comparison
#   5. Generate all figures                           → updated plots
# =============================================================================

echo "=============================================="
echo "Linear Heads Evaluation (Overfitting Fix)"
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

# --- 1. GNN LOMO with linear heads (combined) ---
echo "=========================================="
echo ">>> [1/5] GNN LOMO + linear heads (combined, 143 corners)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --eval-only --mode pretrained --linear-heads --combined --seed 42

# --- 2. GNN LOMO with linear heads (SkillCorner-only) ---
echo ""
echo "=========================================="
echo ">>> [2/5] GNN LOMO + linear heads (SkillCorner-only, 86 corners)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --eval-only --mode pretrained --linear-heads --seed 42

# --- 3. MLP baseline with linear probe (combined) ---
echo ""
echo "=========================================="
echo ">>> [3/5] MLP linear probe (combined, 143 corners)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --baselines mlp --linear-mlp --combined --seed 42

# --- 4. MLP baseline with linear probe (SkillCorner-only) ---
echo ""
echo "=========================================="
echo ">>> [4/5] MLP linear probe (SkillCorner-only, 86 corners)"
echo "=========================================="
python3 -u -m corner_prediction.run_all --baselines mlp --linear-mlp --seed 42

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
