#!/bin/bash
#SBATCH --job-name=phase2_3_sc_temporal
#SBATCH --partition=acltr
#SBATCH --account=researchers
#SBATCH --output=logs/phase2_3_sc_temporal_%j.out
#SBATCH --error=logs/phase2_3_sc_temporal_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Phase 2.3: SkillCorner Temporal Feature Extraction
# Extracts temporal features from SkillCorner tracking data (10fps)
# Creates 5 temporal frames per corner (-2s, -1s, 0s, +1s, +2s)
# Accesses data via GitHub media URLs (bypassing Git LFS)

echo "========================================"
echo "Phase 2.3: SkillCorner Temporal Features"
echo "========================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Navigate to project directory
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies if needed
pip install pandas numpy tqdm --quiet

# Run temporal feature extraction
echo "Extracting temporal features from SkillCorner tracking data..."
python scripts/extract_skillcorner_temporal.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Phase 2.3 Complete!"
    echo "Output: data/features/temporal/skillcorner_temporal_features.parquet"
else
    echo ""
    echo "✗ Phase 2.3 Failed!"
    exit 1
fi

echo ""
echo "End time: $(date)"
