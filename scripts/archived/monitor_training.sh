#!/bin/bash
# Monitor GNN training progress
# Usage: bash scripts/monitor_training.sh [job_id]

JOB_ID=${1:-29233}  # Default to latest job
LOG_FILE="logs/phase3_train_gnn_${JOB_ID}.out"
ERR_FILE="logs/phase3_train_gnn_${JOB_ID}.err"

echo "======================================================================"
echo "Monitoring Training Job: $JOB_ID"
echo "======================================================================"
echo ""

# Check job status
echo "Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not in queue (may have finished or not started)"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "⏳ Waiting for job to start..."
    echo "Log file will appear at: $LOG_FILE"
    echo ""
    echo "To monitor manually:"
    echo "  tail -f $LOG_FILE"
    exit 0
fi

# Show recent log output
echo "Recent Output (last 50 lines):"
echo "----------------------------------------------------------------------"
tail -50 "$LOG_FILE"
echo "----------------------------------------------------------------------"
echo ""

# Check for errors
if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo "⚠️  Errors detected:"
    echo "----------------------------------------------------------------------"
    tail -20 "$ERR_FILE"
    echo "----------------------------------------------------------------------"
    echo ""
fi

# Extract key metrics if available
echo "Training Progress:"
if grep -q "Epoch" "$LOG_FILE"; then
    echo "Current epoch:"
    grep "Epoch" "$LOG_FILE" | tail -1

    if grep -q "AP:" "$LOG_FILE"; then
        echo ""
        echo "Latest validation metrics:"
        grep -A 5 "Val Metrics:" "$LOG_FILE" | tail -6
    fi
else
    echo "Training not started yet..."
fi

echo ""
echo "======================================================================"
echo "Monitoring Commands:"
echo "======================================================================"
echo "Live log:           tail -f $LOG_FILE"
echo "Check status:       squeue -j $JOB_ID"
echo "Cancel job:         scancel $JOB_ID"
echo "View full log:      cat $LOG_FILE"
echo "Check errors:       cat $ERR_FILE"
echo ""
