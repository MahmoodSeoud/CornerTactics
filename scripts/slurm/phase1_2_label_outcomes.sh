#!/bin/bash
#SBATCH --job-name=phase1_2_outcomes
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/phase1_2_outcomes_%j.out
#SBATCH --error=logs/phase1_2_outcomes_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# ============================================================================
# Phase 1.2: Comprehensive Outcome Labeling Pipeline
# ============================================================================
# This script runs the complete Phase 1.2 pipeline:
# 1. Label StatsBomb corners with improved algorithm
# 2. Label SkillCorner corners using dynamic events
# 3. Label SoccerNet corners using Labels JSON files
# 4. Regenerate unified corner database with complete outcomes
# ============================================================================

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo

# Fix libstdc++ compatibility
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

echo "======================================================================"
echo "Phase 1.2: Comprehensive Outcome Labeling Pipeline"
echo "======================================================================"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo ""
echo "Pipeline Steps:"
echo "  1. Label StatsBomb corners (improved algorithm)"
echo "  2. Label SkillCorner corners (dynamic events)"
echo "  3. Label SoccerNet corners (Labels JSON)"
echo "  4. Regenerate unified corner database"
echo ""
echo "======================================================================"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install pandas tqdm statsbombpy numpy --quiet
echo ""

# ============================================================================
# Step 1: Label StatsBomb Corners (Improved Algorithm)
# ============================================================================
echo "======================================================================"
echo "Step 1/4: Labeling StatsBomb Corners with Improved Algorithm"
echo "======================================================================"
echo ""
python scripts/label_statsbomb_outcomes.py
step1_status=$?

if [ $step1_status -ne 0 ]; then
    echo "✗ Step 1 failed with exit code $step1_status"
    exit 1
fi
echo ""

# ============================================================================
# Step 2: Label SkillCorner Corners
# ============================================================================
echo "======================================================================"
echo "Step 2/4: Labeling SkillCorner Corners"
echo "======================================================================"
echo ""
python scripts/label_skillcorner_outcomes.py
step2_status=$?

if [ $step2_status -ne 0 ]; then
    echo "✗ Step 2 failed with exit code $step2_status"
    # Continue even if SkillCorner fails (smaller dataset)
    echo "Continuing to next step..."
fi
echo ""

# ============================================================================
# Step 3: Label SoccerNet Corners
# ============================================================================
echo "======================================================================"
echo "Step 3/4: Labeling SoccerNet Corners"
echo "======================================================================"
echo ""

# Check if SoccerNet data exists
if [ -d "data/datasets/soccernet" ] && [ -f "data/datasets/soccernet/soccernet_corners.csv" ]; then
    echo "SoccerNet corners found, applying labels..."
    python scripts/label_soccernet_outcomes.py
    step3_status=$?

    if [ $step3_status -ne 0 ]; then
        echo "✗ Step 3 failed with exit code $step3_status"
        echo "Continuing to next step..."
    fi
else
    echo "⊘ SoccerNet corners not found, skipping"
    echo "  (Run extract_soccernet_corners.py first if needed)"
    step3_status=0
fi
echo ""

# ============================================================================
# Step 4: Regenerate Unified Database
# ============================================================================
echo "======================================================================"
echo "Step 4/4: Regenerating Unified Corner Database with Outcomes"
echo "======================================================================"
echo ""
python scripts/integrate_corner_datasets.py
step4_status=$?

if [ $step4_status -ne 0 ]; then
    echo "✗ Step 4 failed with exit code $step4_status"
    exit 4
fi
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo "======================================================================"
echo "Phase 1.2 Complete!"
echo "======================================================================"
echo "Completed at: $(date)"
echo ""

echo "Output Files:"
echo "  1. data/datasets/statsbomb/corners_360_with_outcomes.csv"
if [ -f "data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv" ]; then
    echo "  2. data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv"
fi
if [ -f "data/datasets/soccernet/soccernet_corners_with_outcomes.csv" ]; then
    echo "  3. data/datasets/soccernet/soccernet_corners_with_outcomes.csv"
fi
echo "  4. data/unified_corners_dataset.parquet (updated)"
echo "  5. data/unified_corners_dataset.csv (updated)"
echo ""

# Verification
all_good=true

echo "Verification:"
if [ -f "data/datasets/statsbomb/corners_360_with_outcomes.csv" ]; then
    corners=$(wc -l < data/datasets/statsbomb/corners_360_with_outcomes.csv)
    size=$(du -h data/datasets/statsbomb/corners_360_with_outcomes.csv | cut -f1)
    echo "  ✓ StatsBomb outcomes: $corners lines, $size"

    # Check for actual outcomes (not all Possession)
    goals=$(grep -c ",Goal," data/datasets/statsbomb/corners_360_with_outcomes.csv || echo "0")
    shots=$(grep -c ",Shot," data/datasets/statsbomb/corners_360_with_outcomes.csv || echo "0")
    echo "    - Goals detected: $goals"
    echo "    - Shots detected: $shots"

    if [ "$goals" -eq "0" ] && [ "$shots" -eq "0" ]; then
        echo "    ⚠ WARNING: No goals or shots detected - labeling may have failed!"
        all_good=false
    fi
else
    echo "  ✗ Missing: data/datasets/statsbomb/corners_360_with_outcomes.csv"
    all_good=false
fi

if [ -f "data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv" ]; then
    corners=$(wc -l < data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv)
    size=$(du -h data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv | cut -f1)
    echo "  ✓ SkillCorner outcomes: $corners lines, $size"
fi

if [ -f "data/datasets/soccernet/soccernet_corners_with_outcomes.csv" ]; then
    corners=$(wc -l < data/datasets/soccernet/soccernet_corners_with_outcomes.csv)
    size=$(du -h data/datasets/soccernet/soccernet_corners_with_outcomes.csv | cut -f1)
    echo "  ✓ SoccerNet outcomes: $corners lines, $size"
fi

if [ -f "data/unified_corners_dataset.csv" ]; then
    corners=$(wc -l < data/unified_corners_dataset.csv)
    size=$(du -h data/unified_corners_dataset.csv | cut -f1)
    echo "  ✓ Unified dataset (CSV): $corners lines, $size"

    # Check outcome distribution
    echo ""
    echo "Unified Dataset Outcome Distribution:"
    python3 -c "import pandas as pd; df = pd.read_csv('data/unified_corners_dataset.csv'); print(df['outcome_category'].value_counts())" 2>/dev/null || echo "  (Could not analyze outcomes)"
else
    echo "  ✗ Missing: data/unified_corners_dataset.csv"
    all_good=false
fi

echo ""
if [ "$all_good" = true ]; then
    echo "✓✓✓ All Phase 1.2 tasks completed successfully! ✓✓✓"
    echo ""
    echo "Next Steps:"
    echo "  - Review outcome distributions in logs"
    echo "  - Verify goal/shot detection rates are realistic (2-4% goals, 20-25% shots)"
    echo "  - Proceed to Phase 2: Graph Construction"
    exit 0
else
    echo "✗✗✗ Some tasks completed with warnings or errors ✗✗✗"
    echo "Review logs for details"
    exit 7
fi
