#!/bin/bash
#
# Complete Pipeline Execution Script
# Runs all phases in order to reproduce the full CornerTactics dataset and training
#
# Author: mseo
# Date: October 2024
#
# Usage: bash scripts/slurm/RUN_FULL_PIPELINE.sh
#

echo "========================================================================"
echo "CornerTactics Full Pipeline Execution"
echo "========================================================================"
echo "This script will execute all phases to build the complete dataset"
echo "and train the GNN model."
echo ""
echo "Pipeline Overview:"
echo "  Phase 1.1: Data Integration (StatsBomb + SkillCorner)"
echo "  Phase 1.2: Outcome Labeling"
echo "  Phase 2.1: Node Feature Extraction"
echo "  Phase 2.2: Graph Construction (StatsBomb baseline)"
echo "  Phase 2.3: SkillCorner Temporal Extraction"
echo "  Phase 2.4: StatsBomb Temporal Augmentation"
echo "  Phase 2.5: Build SkillCorner Graphs and Merge Datasets"
echo "  Phase 3:   GNN Model Training"
echo ""
echo "Total expected runtime: ~30-45 minutes"
echo "========================================================================"
echo ""

# Function to wait for a job to complete
wait_for_job() {
    local job_id=$1
    local phase_name=$2

    echo "Waiting for job ${job_id} (${phase_name}) to complete..."

    while true; do
        # Check if job is still in queue or running
        job_status=$(squeue -j ${job_id} 2>/dev/null | tail -n +2)

        if [ -z "$job_status" ]; then
            # Job is no longer in queue - it's completed
            echo "✓ Job ${job_id} completed"

            # Check if it succeeded by looking at the log file
            log_file=$(ls -t logs/${phase_name}_${job_id}.out 2>/dev/null | head -1)
            if [ -f "$log_file" ]; then
                if grep -q "Complete\|Success" "$log_file"; then
                    echo "✓ Phase completed successfully"
                    return 0
                else
                    echo "✗ Phase may have failed - check logs/${phase_name}_${job_id}.{out,err}"
                    return 1
                fi
            fi
            return 0
        fi

        sleep 10
    done
}

# Phase 1.1: Data Integration
echo "[1/8] Submitting Phase 1.1: Data Integration..."
JOB_1_1=$(sbatch --parsable scripts/slurm/phase1_1_complete.sh)
echo "Job ID: ${JOB_1_1}"
wait_for_job ${JOB_1_1} "phase1_1_complete"
echo ""

# Phase 1.2: Outcome Labeling
echo "[2/8] Submitting Phase 1.2: Outcome Labeling..."
JOB_1_2=$(sbatch --parsable scripts/slurm/phase1_2_label_outcomes.sh)
echo "Job ID: ${JOB_1_2}"
wait_for_job ${JOB_1_2} "phase1_2_label_outcomes"
echo ""

# Phase 2.1: Node Feature Extraction
echo "[3/8] Submitting Phase 2.1: Node Feature Extraction..."
JOB_2_1=$(sbatch --parsable scripts/slurm/phase2_1_extract_features.sh)
echo "Job ID: ${JOB_2_1}"
wait_for_job ${JOB_2_1} "phase2_1_features"
echo ""

# Phase 2.2: Graph Construction (StatsBomb baseline)
echo "[4/8] Submitting Phase 2.2: Graph Construction..."
JOB_2_2=$(sbatch --parsable scripts/slurm/phase2_2_build_graphs.sh)
echo "Job ID: ${JOB_2_2}"
wait_for_job ${JOB_2_2} "phase2_2_graphs"
echo ""

# Phase 2.3: SkillCorner Temporal Extraction
echo "[5/8] Submitting Phase 2.3: SkillCorner Temporal Extraction..."
JOB_2_3=$(sbatch --parsable scripts/slurm/phase2_3_skillcorner_temporal.sh)
echo "Job ID: ${JOB_2_3}"
wait_for_job ${JOB_2_3} "phase2_3_sc_temporal"
echo ""

# Phase 2.4: StatsBomb Temporal Augmentation
echo "[6/8] Submitting Phase 2.4: StatsBomb Temporal Augmentation..."
JOB_2_4=$(sbatch --parsable scripts/slurm/phase2_4_statsbomb_augment.sh)
echo "Job ID: ${JOB_2_4}"
wait_for_job ${JOB_2_4} "phase2_4_sb_augment"
echo ""

# Phase 2.5: Build SkillCorner Graphs and Merge
echo "[7/8] Submitting Phase 2.5: Build SkillCorner Graphs..."
JOB_2_5=$(sbatch --parsable scripts/slurm/phase2_5_build_skillcorner_graphs.sh)
echo "Job ID: ${JOB_2_5}"
wait_for_job ${JOB_2_5} "phase2_5_sc_graphs"
echo ""

# Phase 3: GNN Training
echo "[8/8] Submitting Phase 3: GNN Training..."
JOB_3=$(sbatch --parsable scripts/slurm/phase3_train_gnn.sh)
echo "Job ID: ${JOB_3}"
echo "Training job submitted. Monitor with: tail -f logs/phase3_train_gnn_${JOB_3}.out"
echo ""

echo "========================================================================"
echo "Pipeline Execution Complete!"
echo "========================================================================"
echo "All phases submitted successfully."
echo ""
echo "Final training job: ${JOB_3}"
echo "  Monitor: tail -f logs/phase3_train_gnn_${JOB_3}.out"
echo "  Check status: squeue -j ${JOB_3}"
echo ""
echo "Expected outputs:"
echo "  - Combined dataset: data/graphs/adjacency_team/combined_temporal_graphs.pkl"
echo "  - Trained model: models/corner_gnn_gcn_shot_<timestamp>/"
echo ""
echo "To check all job statuses: squeue -u \$USER"
echo "========================================================================"
