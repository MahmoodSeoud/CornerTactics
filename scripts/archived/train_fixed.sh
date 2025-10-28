#!/bin/bash
# Quick-start training script with all fixes applied
#
# This script uses the CORRECTED training approach:
# ✓ Weighted BCE loss only (no focal, no balanced sampling)
# ✓ Edge features enabled (6-dim player relationships)
# ✓ Average Precision metric for evaluation
#
# Usage:
#   bash scripts/train_fixed.sh                    # Train with edge features (RECOMMENDED)
#   bash scripts/train_fixed.sh --no-edge          # Train baseline without edge features
#   bash scripts/train_fixed.sh --help             # Show this help

set -e  # Exit on error

# Default settings
MODEL="gcn_edge"
USE_EDGE_FEATURES="--use-edge-features"

# Parse arguments
if [[ "$1" == "--no-edge" ]]; then
    MODEL="gcn"
    USE_EDGE_FEATURES=""
    echo "Training baseline GCN without edge features"
elif [[ "$1" == "--help" ]]; then
    echo "Corner Kick GNN Training - Fixed Version"
    echo ""
    echo "Usage:"
    echo "  bash scripts/train_fixed.sh              # With edge features (recommended)"
    echo "  bash scripts/train_fixed.sh --no-edge    # Without edge features (baseline)"
    echo "  bash scripts/train_fixed.sh --help       # Show this help"
    echo ""
    echo "This script fixes the triple over-correction issue:"
    echo "  ✓ Uses weighted BCE only (not focal loss)"
    echo "  ✓ No balanced batch sampling"
    echo "  ✓ Enables 6-dim edge features (when not using --no-edge)"
    echo "  ✓ Uses Average Precision for model selection"
    echo ""
    echo "Expected results:"
    echo "  - Test AP: 0.35-0.45 (up from 0.176)"
    echo "  - Test Precision: 35-45% (up from 19.5%)"
    echo "  - Test Recall: 65-75% (down from 96.3%, more realistic)"
    echo ""
    exit 0
else
    echo "Training GCN with edge features (RECOMMENDED)"
    echo "Use --no-edge for baseline, or --help for more info"
fi

echo ""
echo "======================================================================"
echo "Corner Kick GNN Training - FIXED VERSION"
echo "======================================================================"
echo ""
echo "Fixes applied:"
echo "  ✓ Using weighted BCE loss (removed focal loss)"
echo "  ✓ No balanced sampling (removed 50/50 batch forcing)"
echo "  ✓ Edge features: $([ -z "$USE_EDGE_FEATURES" ] && echo "DISABLED" || echo "ENABLED (6-dim)")"
echo "  ✓ Using Average Precision (AP) for model selection"
echo ""
echo "Expected improvements:"
echo "  - Test AP: 0.35-0.45 (2-2.5× improvement)"
echo "  - Better calibrated predictions"
echo "  - Realistic precision/recall balance"
echo ""
echo "======================================================================"
echo ""

# Activate conda environment (adjust path if needed)
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'robo'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robo
fi

# Install dependencies if needed
echo "Checking PyTorch Geometric installation..."
python -c "import torch_geometric" 2>/dev/null || {
    echo "Installing PyTorch Geometric..."
    pip install torch torch-geometric pyg-lib torch-scatter torch-sparse --quiet
}

# Run training
echo ""
echo "Starting training..."
echo "Model: $MODEL"
echo ""

python scripts/train_gnn.py \
    --graph-path data/graphs/adjacency_team/combined_temporal_graphs.pkl \
    --outcome-type shot \
    --model "$MODEL" \
    $USE_EDGE_FEATURES \
    --loss weighted \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --patience 15 \
    --dropout 0.3 \
    --weight-decay 1e-4 \
    --scheduler cosine \
    --device cuda \
    --num-workers 4 \
    --save-dir models \
    --log-dir runs

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Training Complete!"
    echo "======================================================================"
    echo ""
    echo "Check results:"
    echo "  - Models saved in: models/"
    echo "  - TensorBoard logs: runs/"
    echo ""
    echo "View training progress:"
    echo "  tensorboard --logdir runs/"
    echo ""
    echo "Key metrics to look for in final output:"
    echo "  - Test AP: Should be 0.35-0.45 (target)"
    echo "  - Test Precision: Should be 35-45%"
    echo "  - Test Recall: Should be 65-75%"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "❌ Training Failed"
    echo "======================================================================"
    echo "Check error messages above for details"
    exit 1
fi
