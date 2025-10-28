#!/bin/bash
#SBATCH --job-name=train_gat
#SBATCH --partition=scavenge
#SBATCH --output=logs/train_gat_%j.out
#SBATCH --error=logs/train_gat_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train GAT model with attention mechanism
# Expected improvement over GCN baseline

echo "============================================================"
echo "Training GAT Model with Attention Mechanism"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
echo "Installing dependencies..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install torch-geometric --quiet
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --quiet
pip install tensorboard scikit-learn tqdm --quiet

# Create directories
mkdir -p logs models runs

# Training configuration
GRAPH_PATH="data/graphs/adjacency_team/combined_temporal_graphs.pkl"
MODEL="gat"
OUTCOME="shot"
EPOCHS=150
BATCH_SIZE=32
LR=0.001
POS_WEIGHT=5.85  # For 17.1% positive class

echo ""
echo "Configuration:"
echo "  Model: GAT (Graph Attention Network)"
echo "  Dataset: 7,369 graphs (1,258 positive)"
echo "  Positive class: 17.1%"
echo "  Pos weight: 5.85"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Node: $SLURMD_NODENAME"
echo ""

# Run training
echo "Starting GAT training..."
python scripts/train_gnn.py \
    --graph-path $GRAPH_PATH \
    --outcome-type $OUTCOME \
    --model $MODEL \
    --loss weighted \
    --pos-weight $POS_WEIGHT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience 20 \
    --scheduler cosine \
    --dropout 0.3 \
    --weight-decay 1e-4 \
    --device cuda \
    --num-workers 4 \
    --save-dir models \
    --log-dir runs

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ GAT Training Complete!"
    echo "============================================================"

    # Show final results
    echo ""
    echo "Extracting final metrics..."
    grep -E "(Test AP:|Test AUC:|Test Accuracy:|Test F1:)" logs/train_gat_${SLURM_JOB_ID}.out | tail -4

    echo ""
    echo "Models saved in: models/corner_gnn_gat_shot_*"
    echo "Tensorboard logs in: runs/"
    echo ""
    echo "Expected improvements over baseline (GCN):"
    echo "  Baseline Test AP: 0.158"
    echo "  Target Test AP: 0.25+ (with GAT attention)"
else
    echo ""
    echo "============================================================"
    echo "❌ GAT Training Failed"
    echo "============================================================"
    echo "Check error log: logs/train_gat_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: ${SECONDS} seconds"