#!/bin/bash
#SBATCH --job-name=dfl_ablation
#SBATCH --output=logs/dfl_ablation_%j.out
#SBATCH --error=logs/dfl_ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1

# DFL Velocity Ablation Experiment
# Tests whether velocity features improve corner kick outcome prediction
# Compares position-only vs position+velocity features

echo "=============================================="
echo "DFL Velocity Ablation Experiment"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="

# Setup environment
cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

# Create directories
mkdir -p logs
mkdir -p results/ablation

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check Python and PyTorch
echo "Python: $(python3 --version)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run ablation experiment
echo "Starting ablation experiment..."
echo ""

python3 scripts/run_ablation.py \
    --data-dir data/dfl \
    --output-dir results/ablation \
    --pretrain-epochs 50 \
    --finetune-epochs 100 \
    --lr 1e-4 \
    --device cuda

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
