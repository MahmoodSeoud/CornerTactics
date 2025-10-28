#!/bin/bash
#SBATCH --job-name=balanced_gnn
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --gres=gpu:1
#SBATCH --output=logs/balanced_gnn_%j.out
#SBATCH --error=logs/balanced_gnn_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

echo "============================================================"
echo "Balanced GNN Training for Corner Kick Prediction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# Activate environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Change to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install torch torch-geometric scikit-learn imbalanced-learn tqdm --quiet

# Step 1: Skip SMOTE for now (optional enhancement)
echo ""
echo "Step 1: Skipping SMOTE augmentation (using original data with balancing techniques)"

# Step 2: Train with balanced techniques
echo ""
echo "Step 2: Training with balanced techniques..."

# Default configuration (balanced approach)
python scripts/train_gnn_balanced.py \
    --loss-type focal \
    --pos-weight 6.0 \
    --use-balanced-sampling \
    --model-type gcn \
    --hidden-dim 64 \
    --num-layers 3 \
    --dropout 0.2 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.001 \
    --weight-decay 1e-4 \
    --outcome-type shot \
    --threshold-optimize \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training complete!"
echo "Results saved to: models/"
echo "End time: $(date)"
echo "============================================================"