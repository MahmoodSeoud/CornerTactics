#!/bin/bash
#SBATCH --job-name=test_receiver_loader
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/test_receiver_loader_%j.out
#SBATCH --error=logs/test_receiver_loader_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Test ReceiverCornerDataset (Day 3-4 Implementation)
# Runs pytest on test_receiver_data_loader.py

echo "=================================="
echo "Testing ReceiverCornerDataset"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project root
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest --quiet

# Run pytest with verbose output
echo ""
echo "Running pytest tests..."
python -m pytest tests/test_receiver_data_loader.py -v -s

# Exit code
EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed!"
fi

exit $EXIT_CODE
