#!/bin/bash
#SBATCH --job-name=exp4_gat
#SBATCH --partition=scavenge
#SBATCH --output=logs/exp4_gat_%j.out
#SBATCH --error=logs/exp4_gat_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Experiment 4: Test GAT (Graph Attention) model
# Uses attention mechanism to focus on dangerous players

echo "============================================================"
echo "Experiment 4: GAT Model with Attention"
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

echo ""
echo "Training GAT model with attention mechanism"
echo "----------------------------------------"
python scripts/train_gnn.py \
    --graph-path data/graphs/adjacency_team/combined_temporal_graphs.pkl \
    --outcome-type shot \
    --model gat \
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
    echo "✅ Experiment 4 Complete"
else
    echo ""
    echo "❌ Experiment 4 Failed"
    exit 1
fi
