#!/bin/bash
#SBATCH --job-name=random-init-gnn
#SBATCH --output=logs/random_init_gnn_%j.out
#SBATCH --error=logs/random_init_gnn_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=28G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=acltr
#SBATCH --account=students

# Random-init GNN: same USSF architecture, no pretrained weights, all trainable.
# Runs 5-seed evaluation + 100-permutation test on combined dataset (143 corners).

set -euo pipefail

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

mkdir -p logs
export PYTHONPATH=/home/mseo/CornerTactics:${PYTHONPATH:-}

echo "=== Random-Init GNN Experiment ==="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi || echo "No GPU info available"

# Run multi-seed evaluation + permutation test
python scripts/random_init_gnn.py \
    --multi-seed \
    --permutation \
    --combined \
    --n-permutations 100

echo "Finished: $(date)"
