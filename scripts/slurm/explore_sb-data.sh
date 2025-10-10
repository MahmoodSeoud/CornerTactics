#!/bin/bash
#SBATCH --job-name=sb_explore
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/sb_explore_%j.out
#SBATCH --error=logs/sb_explore_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# Load conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Set work directory
cd /home/mseo/CornerTactics

# Add project root to Python path for src module imports
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "Starting StatsBomb data exploration at $(date)"
echo "Node: $(hostname)"

# Create output directories
mkdir -p logs
mkdir -p data/statsbomb

# Install statsbombpy if not already installed
echo ""
echo "Checking statsbombpy installation..."
if ! python -c "import statsbombpy" 2>/dev/null; then
    echo "Installing statsbombpy..."
    pip install statsbombpy --quiet
    echo "statsbombpy installed successfully"
else
    echo "statsbombpy already installed"
fi

# Run exploration script
echo ""
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo
echo "Exploring StatsBomb Open Data"
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo
echo ""

python scripts/explore_statsbomb.py

EXITCODE=$?

echo ""
echo "Exploration completed at $(date)"
echo "Exit code: $EXITCODE"

# Report results
echo ""
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo
echo "RESULTS"
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo

if [ $EXITCODE -eq 0 ]; then
    echo "✓ Exploration completed successfully"
    echo ""

    # Check for output files
    if [ -f "data/statsbomb/available_competitions.csv" ]; then
        echo "✓ Competitions list: data/statsbomb/available_competitions.csv"
        COMP_COUNT=$(tail -n +2 data/statsbomb/available_competitions.csv | wc -l)
        echo "  → $COMP_COUNT competitions found"
    else
        echo "⚠ Competitions list not created"
    fi

    if [ -f "data/statsbomb/sample_corner_events.csv" ]; then
        echo "✓ Sample corners: data/statsbomb/sample_corner_events.csv"
        CORNER_COUNT=$(tail -n +2 data/statsbomb/sample_corner_events.csv | wc -l)
        echo "  → $CORNER_COUNT corner kicks extracted"

        # Show outcome distribution
        echo ""
        echo "Outcome distribution:"
        tail -n +2 data/statsbomb/sample_corner_events.csv | \
            cut -d',' -f13 | \
            sort | uniq -c | sort -rn | \
            head -10 | \
            sed 's/^/  /'
    else
        echo "⚠ Sample corners not created"
    fi

    echo ""
    echo "Storage used:"
    du -sh data/statsbomb 2>/dev/null || echo "  (directory not found)"

else
    echo "✗ Exploration failed (exit code: $EXITCODE)"
    echo ""
    echo "Check logs for errors:"
    echo "  - logs/sb_explore_${SLURM_JOB_ID}.out"
    echo "  - logs/sb_explore_${SLURM_JOB_ID}.err"
fi

echo ""
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo
echo "Next steps:"
echo "  1. Review: data/statsbomb/available_competitions.csv"
echo "  2. Review: data/statsbomb/sample_corner_events.csv"
echo "  3. Select competitions for full dataset extraction"
echo "  4. Build corner outcome prediction dataset"
echo "=" | tr -d '\n'; printf '=%.0s' {1..69}; echo

exit $EXITCODE
