#!/bin/bash
#SBATCH --job-name=exp2_distance
#SBATCH --partition=scavenge
#SBATCH --output=logs/exp2_distance_%j.out
#SBATCH --error=logs/exp2_distance_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Experiment 2: Test distance-based adjacency strategy
# Connect players within 10m radius

echo "============================================================"
echo "Experiment 2: Distance-Based Adjacency"
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
pip install tensorboard scikit-learn tqdm scipy --quiet

# Create necessary directories
mkdir -p logs models runs data/graphs/adjacency_distance

echo ""
echo "Step 1: Build distance-based graph dataset"
echo "----------------------------------------"
python scripts/build_skillcorner_graphs.py \
    --strategy distance \
    --output-dir data/graphs/adjacency_distance

if [ $? -ne 0 ]; then
    echo "❌ Failed to build distance graphs"
    exit 1
fi

echo ""
echo "Step 2: Train GNN with distance adjacency"
echo "----------------------------------------"
python scripts/train_gnn.py \
    --graph-path data/graphs/adjacency_distance/combined_temporal_graphs.pkl \
    --outcome-type shot \
    --model gcn \
    --loss weighted \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --patience 15 \
    --scheduler cosine \
    --dropout 0.3 \
    --weight-decay 1e-4 \
    --device cuda \
    --num-workers 4 \
    --save-dir models \
    --log-dir runs

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiment 2 Complete"
else
    echo ""
    echo "❌ Experiment 2 Failed"
    exit 1
fi
