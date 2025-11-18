#!/bin/bash
# Master script to run the complete retraining pipeline with SLURM job dependencies
#
# Pipeline:
# 1. Extract features (49 features) - CPU job
# 2. Train models - GPU job (depends on step 1)
# 3. Evaluate models - CPU job (depends on step 2)

set -e  # Exit on error

echo "=========================================="
echo "Corner Kick Prediction Retraining Pipeline"
echo "49 Features | 3 Model Types | Full Evaluation"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Step 1: Submit feature extraction job
echo "Step 1: Submitting feature extraction job..."
JOB1=$(sbatch --parsable slurm_extract_features.sh)
echo "  Job ID: $JOB1"
echo ""

# Step 2: Submit training job (depends on feature extraction)
echo "Step 2: Submitting model training job (depends on Job $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_train_baseline.sh)
echo "  Job ID: $JOB2"
echo ""

# Step 3: Submit evaluation job (depends on training)
echo "Step 3: Submitting evaluation job (depends on Job $JOB2)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm_evaluate.sh)
echo "  Job ID: $JOB3"
echo ""

echo "=========================================="
echo "Pipeline submitted successfully!"
echo "=========================================="
echo ""
echo "Job Dependencies:"
echo "  $JOB1 (extract) -> $JOB2 (train) -> $JOB3 (evaluate)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "View logs in real-time:"
echo "  tail -f logs/extract_features_${JOB1}.out"
echo "  tail -f logs/train_baseline_${JOB2}.out"
echo "  tail -f logs/evaluate_${JOB3}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB1 $JOB2 $JOB3"
echo ""
