#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=scavenge
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo "Quick throwaway test - Random Forest baseline"
echo "============================================================"

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Just run with basic dependencies
python scripts/quick_train.py

echo ""
echo "============================================================"
echo "Done! Check output above for results"