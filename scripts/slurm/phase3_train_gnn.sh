#!/bin/bash
#SBATCH --job-name=phase3_train_gnn
#SBATCH --partition=acltr
#SBATCH --output=logs/phase3_train_gnn_%j.out
#SBATCH --error=logs/phase3_train_gnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_40gb:1

# Phase 3: Train GNN Model for Corner Kick Prediction
# Trains a Graph Neural Network on corner kick data using PyTorch Geometric

echo "============================================================"
echo "Phase 3: GNN Model Training"
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

# Install PyTorch and PyTorch Geometric dependencies
echo "Installing dependencies..."
echo "Note: This may take a few minutes on first run..."

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet

# Install PyTorch Geometric and related packages
pip install torch-geometric --quiet
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --quiet

# Install other dependencies
pip install tensorboard scikit-learn tqdm --quiet

# Create necessary directories
mkdir -p logs models runs

# Training parameters (can be overridden with environment variables)
MODEL=${MODEL:-gcn}              # Model type: gcn or gat
STRATEGY=${STRATEGY:-team}       # Adjacency strategy
EPOCHS=${EPOCHS:-100}            # Number of epochs
BATCH_SIZE=${BATCH_SIZE:-32}    # Batch size
LR=${LR:-0.001}                  # Learning rate
PATIENCE=${PATIENCE:-15}         # Early stopping patience
LOSS=${LOSS:-weighted}           # Loss function: bce, weighted, or focal

echo ""
echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Adjacency Strategy: $STRATEGY"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Loss Function: $LOSS"
echo ""

# Use combined temporal dataset (StatsBomb + SkillCorner)
GRAPH_PATH="data/graphs/adjacency_${STRATEGY}/combined_temporal_graphs.pkl"

# Check if graph dataset exists
if [ ! -f "$GRAPH_PATH" ]; then
    echo "Combined temporal graph dataset not found at $GRAPH_PATH"
    echo "Please run Phase 2.5 first: sbatch scripts/slurm/phase2_5_build_skillcorner_graphs.sh"
    echo "Or run full pipeline: bash scripts/slurm/RUN_FULL_PIPELINE.sh"
    exit 1
fi

echo "Using combined temporal dataset: $GRAPH_PATH"
echo "Expected: 7,369 graphs (5,814 StatsBomb + 1,555 SkillCorner)"
echo ""

# Run training
echo "Starting GNN training..."
python scripts/train_gnn.py \
    --graph-path $GRAPH_PATH \
    --outcome-type shot \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --loss $LOSS \
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
    echo "✅ Phase 3 Training Complete"
    echo "============================================================"
    echo ""
    echo "Models saved in: models/"
    echo "Tensorboard logs in: runs/"
    echo ""
    echo "To view training progress:"
    echo "  tensorboard --logdir runs/"
    echo ""
    echo "Next Steps:"
    echo "  1. Evaluate model performance: sbatch scripts/slurm/phase3_evaluate_gnn.sh"
    echo "  2. Run ablation studies with different adjacency strategies"
    echo "  3. Analyze feature importance and spatial patterns"
else
    echo ""
    echo "============================================================"
    echo "❌ Phase 3 Training Failed"
    echo "============================================================"
    echo "Check error log: logs/phase3_train_gnn_${SLURM_JOB_ID}.err"
    exit 1
fi

# Print final summary
echo ""
echo "============================================================"
echo "Training Summary"
echo "============================================================"
echo "Job completed at: $(date)"
echo "Total runtime: ${SECONDS} seconds"

# List created model files
echo ""
echo "Created model files:"
ls -lh models/corner_gnn_${MODEL}_goal_*/ 2>/dev/null | tail -5
