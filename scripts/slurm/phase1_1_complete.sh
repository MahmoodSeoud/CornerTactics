#!/bin/bash
#SBATCH --job-name=phase1_1_full
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/phase1_1_full_%j.out
#SBATCH --error=logs/phase1_1_full_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# ============================================================================
# Phase 1.1: Complete Data Integration Pipeline
# ============================================================================
# This script runs the complete Phase 1.1 pipeline:
# 1. Download StatsBomb 360 corners
# 2. Download SoccerNet data (videos + labels + tracking)
# 3. Extract SoccerNet corner clips
# 4. Extract SkillCorner corners
# 5. Label StatsBomb outcomes
# 6. Create unified corner database
# ============================================================================

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Fix libstdc++ compatibility - use conda's libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# SoccerNet password
SOCCERNET_PASSWORD="s0cc3rn3t"

echo "======================================================================"
echo "Phase 1.1: Complete Data Integration & Enrichment Pipeline"
echo "======================================================================"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Pipeline Steps:"
echo "  1. Download StatsBomb 360 corners"
echo "  2. Download SoccerNet data"
echo "  3. Extract SoccerNet corner clips"
echo "  4. Extract SkillCorner corners"
echo "  5. Label StatsBomb outcomes"
echo "  6. Create unified corner database"
echo ""
echo "======================================================================"
echo ""

# Install all dependencies upfront
echo "Installing dependencies..."
pip install pandas tqdm statsbombpy pyarrow ffmpeg-python --quiet
echo ""

# ============================================================================
# Step 1: Download StatsBomb 360 Corners
# ============================================================================
echo "======================================================================"
echo "Step 1/6: Downloading StatsBomb 360 Corner Data"
echo "======================================================================"
echo ""
python scripts/download_statsbomb_corners.py
step1_status=$?

if [ $step1_status -ne 0 ]; then
    echo "✗ Step 1 failed with exit code $step1_status"
    echo "Continuing to next steps..."
fi
echo ""

# ============================================================================
# Step 2: Download SoccerNet Data (Optional - takes very long)
# ============================================================================
echo "======================================================================"
echo "Step 2/6: Downloading SoccerNet Data (OPTIONAL)"
echo "======================================================================"
echo ""
echo "NOTE: This step downloads videos and can take 2-3 days."
echo "Skipping by default. To enable, uncomment in script."
echo ""

# Uncomment to enable SoccerNet download:
# echo "Starting SoccerNet download..."
# python -c "from SoccerNet.Downloader import SoccerNetDownloader as SNdl
# mySNdl = SNdl(LocalDirectory='data/raw/soccernet')
# mySNdl.downloadGames(files=['Labels-v2.json', 'Labels-v3.json'], split=['train', 'valid', 'test'], password='$SOCCERNET_PASSWORD')
# mySNdl.downloadGames(files=['1_720p.mkv', '2_720p.mkv'], split=['train', 'valid', 'test'], password='$SOCCERNET_PASSWORD')
# mySNdl.downloadGames(files=['gameinfo.ini'], split=['train', 'valid', 'test'], password='$SOCCERNET_PASSWORD')"
# step2_status=$?

step2_status=0  # Skip by default
echo "✓ Step 2 skipped (enable in script if needed)"
echo ""

# ============================================================================
# Step 3: Extract SoccerNet Corner Clips
# ============================================================================
echo "======================================================================"
echo "Step 3/6: Extracting SoccerNet Corner Clips"
echo "======================================================================"
echo ""

# Check if SoccerNet data exists
if [ -d "data/raw/soccernet" ] && [ "$(find data/raw/soccernet -name 'Labels*.json' | head -1)" ]; then
    echo "SoccerNet data found, extracting corner clips..."
    python scripts/extract_soccernet_corners.py \
        --data-dir data/raw/soccernet \
        --output data/raw/soccernet/soccernet_corners.csv \
        --split-by-visibility
    step3_status=$?

    if [ $step3_status -ne 0 ]; then
        echo "✗ Step 3 failed with exit code $step3_status"
    fi
else
    echo "⊘ SoccerNet data not found, skipping corner extraction"
    echo "  (Run Step 2 first or manually download SoccerNet data)"
    step3_status=0  # Not an error, just skip
fi
echo ""

# ============================================================================
# Step 4: Extract SkillCorner Corners
# ============================================================================
echo "======================================================================"
echo "Step 4/6: Extracting SkillCorner Corner Events"
echo "======================================================================"
echo ""
python scripts/extract_skillcorner_corners.py
step4_status=$?

if [ $step4_status -ne 0 ]; then
    echo "✗ Step 4 failed with exit code $step4_status"
    exit 4
fi
echo ""

# ============================================================================
# Step 5: Label StatsBomb Outcomes
# ============================================================================
echo "======================================================================"
echo "Step 5/6: Labeling StatsBomb Corner Outcomes"
echo "======================================================================"
echo ""
python scripts/label_statsbomb_outcomes.py
step5_status=$?

if [ $step5_status -ne 0 ]; then
    echo "✗ Step 5 failed with exit code $step5_status"
    exit 5
fi
echo ""

# ============================================================================
# Step 6: Create Unified Database
# ============================================================================
echo "======================================================================"
echo "Step 6/6: Creating Unified Corner Database"
echo "======================================================================"
echo ""
python scripts/integrate_corner_datasets.py
step6_status=$?

if [ $step6_status -ne 0 ]; then
    echo "✗ Step 6 failed with exit code $step6_status"
    exit 6
fi
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo "======================================================================"
echo "Phase 1.1 Complete!"
echo "======================================================================"
echo "Completed at: $(date)"
echo ""
echo "Output Files:"
echo "  1. data/raw/statsbomb/corners_360.csv"
echo "  2. data/raw/statsbomb/corners_360_with_outcomes.csv"
echo "  3. data/raw/skillcorner/skillcorner_corners.csv"
if [ -f "data/raw/soccernet/soccernet_corners.csv" ]; then
    echo "  4. data/raw/soccernet/soccernet_corners.csv"
fi
echo "  5. data/processed/unified_corners_dataset.parquet"
echo "  6. data/processed/unified_corners_dataset.csv"
echo ""

# Verify all critical files exist
all_good=true

echo "Verification:"
if [ -f "data/raw/statsbomb/corners_360.csv" ]; then
    echo "  ✓ StatsBomb corners: $(wc -l < data/raw/statsbomb/corners_360.csv) lines, $(du -h data/raw/statsbomb/corners_360.csv | cut -f1)"
else
    echo "  ✗ Missing: data/raw/statsbomb/corners_360.csv"
    all_good=false
fi

if [ -f "data/raw/skillcorner/skillcorner_corners.csv" ]; then
    echo "  ✓ SkillCorner corners: $(wc -l < data/raw/skillcorner/skillcorner_corners.csv) lines, $(du -h data/raw/skillcorner/skillcorner_corners.csv | cut -f1)"
else
    echo "  ✗ Missing: data/raw/skillcorner/skillcorner_corners.csv"
    all_good=false
fi

if [ -f "data/raw/statsbomb/corners_360_with_outcomes.csv" ]; then
    echo "  ✓ StatsBomb outcomes: $(wc -l < data/raw/statsbomb/corners_360_with_outcomes.csv) lines, $(du -h data/raw/statsbomb/corners_360_with_outcomes.csv | cut -f1)"
else
    echo "  ✗ Missing: data/raw/statsbomb/corners_360_with_outcomes.csv"
    all_good=false
fi

if [ -f "data/processed/unified_corners_dataset.parquet" ]; then
    echo "  ✓ Unified dataset (parquet): $(du -h data/processed/unified_corners_dataset.parquet | cut -f1)"
else
    echo "  ✗ Missing: data/processed/unified_corners_dataset.parquet"
    all_good=false
fi

if [ -f "data/processed/unified_corners_dataset.csv" ]; then
    echo "  ✓ Unified dataset (CSV): $(du -h data/processed/unified_corners_dataset.csv | cut -f1)"
else
    echo "  ✗ Missing: data/processed/unified_corners_dataset.csv"
    all_good=false
fi

if [ -f "data/raw/soccernet/soccernet_corners.csv" ]; then
    echo "  ✓ SoccerNet corners: $(wc -l < data/raw/soccernet/soccernet_corners.csv) lines, $(du -h data/raw/soccernet/soccernet_corners.csv | cut -f1)"
fi

echo ""
if [ "$all_good" = true ]; then
    echo "✓✓✓ All critical Phase 1.1 tasks completed successfully! ✓✓✓"
    exit 0
else
    echo "✗✗✗ Some output files are missing ✗✗✗"
    exit 7
fi
