#!/bin/bash
#SBATCH --job-name=phase5-dfl
#SBATCH --output=../logs/phase5_%j.out
#SBATCH --error=../logs/phase5_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge

# Phase 5: DFL Open-Play Pretraining Comparison
#
# This job extracts open-play samples from 7 DFL matches, trains a backbone
# on shot prediction, and compares transfer learning performance against USSF.
#
# Usage:
#   sbatch phase5_job.sbatch                    # Full pipeline
#   sbatch --export=STAGE=extract phase5_job.sbatch    # Extraction only
#   sbatch --export=STAGE=train phase5_job.sbatch      # Training only
#   sbatch --export=STAGE=transfer phase5_job.sbatch   # Transfer only

set -e

cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Create logs directory if needed
mkdir -p transfer_learning/logs

echo "=============================================="
echo "Phase 5: DFL Open-Play Pretraining Comparison"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Stage: ${STAGE:-full}"
echo ""

# Stage selection
case "${STAGE:-full}" in
    extract)
        echo "Running extraction only..."
        python -u transfer_learning/phase5_dfl_openplay_comparison.py --extract-only
        ;;
    train)
        echo "Running backbone training only..."
        python -u transfer_learning/phase5_dfl_openplay_comparison.py \
            --skip-extraction \
            --train-backbone-only \
            --epochs-pretrain 100
        ;;
    transfer)
        echo "Running transfer experiments only..."
        python -u transfer_learning/phase5_dfl_openplay_comparison.py \
            --skip-extraction \
            --skip-backbone \
            --epochs-finetune 50
        ;;
    full|*)
        echo "Running full pipeline..."
        python -u transfer_learning/phase5_dfl_openplay_comparison.py \
            --epochs-pretrain 100 \
            --epochs-finetune 50
        ;;
esac

echo ""
echo "Completed at: $(date)"
echo "=============================================="
