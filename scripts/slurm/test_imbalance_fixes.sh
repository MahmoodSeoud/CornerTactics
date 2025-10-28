#!/bin/bash
#SBATCH --job-name=exp1_imbalance
#SBATCH --partition=scavenge
#SBATCH --output=logs/exp1_imbalance_%j.out
#SBATCH --error=logs/exp1_imbalance_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Experiment 1: Test class imbalance handling strategies
# - Merge Goal into Shot class
# - Apply SMOTE oversampling
# - Use focal loss + balanced batches

echo "============================================================"
echo "Experiment 1: Class Imbalance Handling"
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
pip install tensorboard scikit-learn tqdm imbalanced-learn --quiet

# Create necessary directories
mkdir -p logs models runs data/graphs/adjacency_team_merged

echo ""
echo "Step 1: Merge Goal class into Shot class"
echo "----------------------------------------"
python scripts/merge_goal_into_shot.py \
    --input data/graphs/adjacency_team/combined_temporal_graphs.pkl \
    --output data/graphs/adjacency_team_merged/combined_temporal_merged.pkl

if [ $? -ne 0 ]; then
    echo "❌ Failed to merge labels"
    exit 1
fi

echo ""
echo "Step 2: Apply SMOTE oversampling"
echo "----------------------------------------"
python scripts/apply_smote.py \
    --input data/graphs/adjacency_team_merged/combined_temporal_merged.pkl \
    --output data/graphs/adjacency_team_merged/combined_temporal_smote.pkl \
    --strategy 0.5

if [ $? -ne 0 ]; then
    echo "❌ Failed to apply SMOTE"
    exit 1
fi

echo ""
echo "Step 3: Train with focal loss + balanced batches"
echo "----------------------------------------"
python scripts/train_gnn.py \
    --graph-path data/graphs/adjacency_team_merged/combined_temporal_smote.pkl \
    --outcome-type shot \
    --model gcn \
    --loss focal \
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
    echo "✅ Experiment 1 Complete"
else
    echo ""
    echo "❌ Experiment 1 Failed"
    exit 1
fi
