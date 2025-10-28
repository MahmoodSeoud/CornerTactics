#!/bin/bash
#SBATCH --job-name=inspect_receiver
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/inspect_receiver_%j.out
#SBATCH --error=logs/inspect_receiver_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

echo "Inspecting receiver graph structure"
echo "Job ID: $SLURM_JOB_ID"

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

python scripts/inspect_receiver_graph.py
