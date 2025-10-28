#!/bin/bash
#SBATCH --job-name=test_ff
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/test_ff_%j.out
#SBATCH --error=logs/test_ff_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "Testing StatsBomb freeze frame data structure"
echo "Job ID: $SLURM_JOB_ID"

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

pip install statsbombpy --quiet

python scripts/test_statsbomb_freeze_frame.py
