#!/bin/bash
#SBATCH --job-name=ms_exp
#SBATCH --output=logs/ms_exp_%A_%a.out
#SBATCH --error=logs/ms_exp_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Multi-Source Corner Kick Prediction Experiments
# Submit as array job: sbatch --array=1-5 scripts/slurm/run_multi_source_exp.sbatch
#
# Array tasks:
#   1 = Experiment 1: Full dataset (4 conditions × 5 seeds + permutation tests)
#   2 = Experiment 2: Source-stratified analysis (5 seeds)
#   3 = Experiment 3: Source ablation (3 conditions × 5 seeds)
#   4 = Experiment 4: Feature ablation (2 conditions × 5 seeds + permutation tests)
#   5 = Experiment 5: kNN adjacency (4 conditions × 5 seeds + permutation tests)

echo "=============================================="
echo "Multi-Source Experiment: Task $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "=============================================="

cd /home/mseo/CornerTactics
export PYTHONPATH=/home/mseo/CornerTactics:$PYTHONPATH

mkdir -p logs
mkdir -p results/multi_source_experiments

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

COMMON_ARGS="--seeds 42 123 456 789 1234 --batch-size 32"

case $SLURM_ARRAY_TASK_ID in
    1)
        echo ">>> Experiment 1: Full Multi-Source Dataset"
        python3 -u transfer_learning/exp1_full_dataset.py \
            $COMMON_ARGS --epochs 100 --patience 15 --n-permutations 20
        ;;
    2)
        echo ">>> Experiment 2: Source-Stratified Analysis"
        python3 -u transfer_learning/exp2_source_stratified.py \
            $COMMON_ARGS --epochs 100 --patience 15
        ;;
    3)
        echo ">>> Experiment 3: Source Ablation"
        python3 -u transfer_learning/exp3_source_ablation.py \
            $COMMON_ARGS --epochs 100 --patience 15
        ;;
    4)
        echo ">>> Experiment 4: Feature Ablation"
        python3 -u transfer_learning/exp4_feature_ablation.py \
            $COMMON_ARGS --epochs 100 --patience 15 --n-permutations 20
        ;;
    5)
        echo ">>> Experiment 5: kNN Adjacency (k=5)"
        python3 -u transfer_learning/exp5_knn_adjacency.py \
            $COMMON_ARGS --epochs 100 --patience 15 --n-permutations 20 --k 5
        ;;
    *)
        echo "ERROR: Unknown task ID $SLURM_ARRAY_TASK_ID (expected 1-5)"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Completed: $(date) (exit code: $EXIT_CODE)"
echo "=============================================="
