#!/bin/bash
# Monitor SLURM download jobs

echo "=== SLURM Job Status ==="
squeue -u $USER

echo -e "\n=== Available Log Files ==="
ls -la logs/

echo -e "\n=== Quick Commands ==="
echo "Watch 224p progress: tail -f logs/download_19783.err"
echo "Watch 720p progress: tail -f logs/download_720p_*.err"
echo "Check data folder:   ls -la data/"
echo "Job status:          squeue -u $USER"

# Show recent progress if logs exist
if ls logs/download_*.err 1> /dev/null 2>&1; then
    echo -e "\n=== Recent Progress (last 10 lines) ==="
    tail -10 logs/download_*.err | tail -10
fi