#!/bin/bash
# Train GAT model with improved settings
# Using attention mechanism to identify important player connections

echo "============================================================"
echo "Training GAT Model with Attention Mechanism"
echo "============================================================"
echo ""

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Create directories
mkdir -p models logs runs

# Training configuration
GRAPH_PATH="data/graphs/adjacency_team/combined_temporal_graphs.pkl"
MODEL="gat"
OUTCOME="shot"
EPOCHS=150
BATCH_SIZE=32
LR=0.001
POS_WEIGHT=5.85  # For 17.1% positive class

echo "Configuration:"
echo "  Model: GAT (Graph Attention Network)"
echo "  Dataset: 7,369 graphs (1,258 positive)"
echo "  Positive class: 17.1%"
echo "  Pos weight: 5.85"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install torch torch-geometric tensorboard scikit-learn tqdm --quiet 2>/dev/null

echo "Starting training..."
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

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Training Complete!"
    echo "============================================================"
    echo ""
    echo "Check results in:"
    echo "  Models: models/corner_gnn_gat_shot_*/"
    echo "  Logs: runs/"
    echo ""
    echo "To view training progress:"
    echo "  tensorboard --logdir runs/"
else
    echo ""
    echo "❌ Training failed!"
    echo "Check error output above"
fi