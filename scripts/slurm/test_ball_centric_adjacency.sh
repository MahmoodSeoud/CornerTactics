#!/bin/bash
#SBATCH --job-name=exp3_ballcentric
#SBATCH --partition=scavenge
#SBATCH --output=logs/exp3_ballcentric_%j.out
#SBATCH --error=logs/exp3_ballcentric_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Experiment 3: Test ball-centric adjacency strategy
# Focus on connections near ball landing zone

echo "============================================================"
echo "Experiment 3: Ball-Centric Adjacency"
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
mkdir -p logs models runs data/graphs/adjacency_ball_centric

echo ""
echo "Step 1: Build ball-centric graph dataset"
echo "----------------------------------------"
python scripts/build_skillcorner_graphs.py \
    --strategy ball_centric \
    --output-dir data/graphs/adjacency_ball_centric

if [ $? -ne 0 ]; then
    echo "❌ Failed to build ball-centric graphs"
    exit 1
fi

echo ""
echo "Step 2: Train GNN with ball-centric adjacency"
echo "----------------------------------------"
python scripts/train_gnn.py \
    --graph-path data/graphs/adjacency_ball_centric/combined_temporal_graphs.pkl \
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
    echo "✅ Experiment 3 Complete"
else
    echo ""
    echo "❌ Experiment 3 Failed"
    exit 1
fi
