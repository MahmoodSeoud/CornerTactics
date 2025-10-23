# Phase 1.2: Outcome Labeling Pipeline - Implementation Summary

**Status**: Ready for execution
**Date**: October 22, 2025
**Author**: Claude Code

## Overview

Phase 1.2 implements comprehensive outcome labeling for all 5,643 corner kicks across three data sources (StatsBomb, SkillCorner, SoccerNet), addressing the critical gap identified in Phase 1.1 where all corners were incorrectly labeled as "Possession".

## Problem Statement

Phase 1.1 created a unified corner dataset but failed to properly label outcomes:
- **StatsBomb (1,118 corners)**: All labeled as "Possession", ZERO goals/shots detected
- **SkillCorner (317 corners)**: No outcome labels
- **SoccerNet (4,208 corners)**: No outcome labels

This made the dataset unsuitable for outcome prediction modeling in Phase 2.

## Solution Architecture

### Core Module: `src/outcome_labeler.py`

Unified outcome labeling framework with three specialized labelers:

1. **`StatsBombOutcomeLabeler`**
   - Scans events following corner kicks (up to 20 seconds)
   - Priority-based classification: Goal > Shot > Clearance > Second Corner > Loss > Possession
   - Extracts temporal features (time_to_outcome, events_to_outcome)
   - Calculates xThreat delta (expected threat change)

2. **`SkillCornerOutcomeLabeler`**
   - Parses dynamic_events.csv files
   - Uses continuous tracking data (10fps)
   - Links to phases_of_play.csv for possession context
   - Handles frame-based timing (10 fps conversion)

3. **`SoccerNetOutcomeLabeler`**
   - Parses Labels-v2.json and Labels-v3.json files
   - Matches annotations to corner timestamps
   - Supports SoccerNet time format ("1 - 12:34")
   - Falls back gracefully when labels missing

### Outcome Classification

**Primary Categories:**
- **Goal**: Goal scored within 20 seconds
- **Shot**: Shot attempt (saved/blocked/off target/post)
- **Clearance**: Defensive clearance by defending team
- **Possession**: Maintained possession or second corner
- **Loss**: Interception, foul won, or duel lost

**Metrics Calculated:**
- `outcome_category`: High-level classification
- `outcome_type`: Detailed event type
- `time_to_outcome`: Seconds from corner to outcome
- `events_to_outcome`: Number of events between
- `goal_scored`: Boolean flag
- `xthreat_delta`: Change in expected threat
- `outcome_location`: (x, y) coordinates of outcome

## Implementation Files

### Core Modules
- `src/outcome_labeler.py` (670 lines) - Unified labeling framework

### Labeling Scripts
- `scripts/label_statsbomb_outcomes_v2.py` - StatsBomb labeler (fixed algorithm)
- `scripts/label_skillcorner_outcomes.py` - SkillCorner labeler
- `scripts/label_soccernet_outcomes.py` - SoccerNet labeler

### Integration
- `scripts/integrate_corner_datasets.py` (updated) - Loads outcome-labeled versions
- `scripts/slurm/phase1_2_label_outcomes.sh` - Master pipeline script

### Testing
- `scripts/test_outcome_detection.py` - Debugging test (throwaway)
- `scripts/slurm/test_outcome_detection.sh` - SLURM test job

## Expected Results

### Success Metrics

**Goal Detection Rate**: 2-4% of corners
(Realistic rate based on soccer statistics - corners rarely result in goals)

**Shot Detection Rate**: 20-25% of corners
(Includes goals + shot attempts)

**Clearance Rate**: 30-40% of corners
(Most common defensive outcome)

**Possession Rate**: 30-40%
(Retained possession without immediate shot/clearance)

### Output Files

1. `data/datasets/statsbomb/corners_360_with_outcomes.csv` (1,118 corners)
2. `data/datasets/skillcorner/skillcorner_corners_with_outcomes.csv` (317 corners)
3. `data/datasets/soccernet/soccernet_corners_with_outcomes.csv` (4,208 corners)
4. `data/unified_corners_dataset.parquet` (5,643 corners, updated)
5. `data/unified_corners_dataset.csv` (5,643 corners, updated)

## Execution Instructions

### Submit Phase 1.2 Pipeline

```bash
sbatch scripts/slurm/phase1_2_label_outcomes.sh
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# View logs in real-time
tail -f logs/phase1_2_outcomes_<job_id>.out
tail -f logs/phase1_2_outcomes_<job_id>.err
```

### Estimated Runtime

- StatsBomb labeling: 2-3 hours (API calls for ~50 matches)
- SkillCorner labeling: 10-15 minutes (10 matches, local files)
- SoccerNet labeling: 1-2 hours (4,208 corners, JSON parsing)
- Integration: 5-10 minutes
- **Total**: ~4-6 hours

## Validation Checks

The SLURM script includes automated validation:

1. **File Existence**: All output files created
2. **Goal Detection**: Goals > 0 (not all "Possession")
3. **Shot Detection**: Shots > 0
4. **Outcome Distribution**: Printed to logs
5. **Dataset Merge**: Unified dataset regenerated successfully

## Key Improvements Over Phase 1.1

1. **Proper Event Detection**: Fixed StatsBomb matching logic
   - Added `pass_type='Corner'` filter
   - Improved timestamp-based matching
   - Extended search window (30 events vs 20)

2. **Priority-Based Classification**: Ensures logical outcome hierarchy

3. **xThreat Calculation**: Quantifies threat change from corner to outcome

4. **Temporal Features**: time_to_outcome and events_to_outcome

5. **Unified Framework**: Consistent classification across all data sources

6. **Graceful Degradation**: Falls back to base datasets if outcomes unavailable

## Known Limitations

1. **SkillCorner**: Limited event types in dynamic_events.csv
   - May miss some shots/goals
   - Consider manual validation for research use

2. **SoccerNet**: Label coverage varies by match
   - Some matches may lack detailed annotations
   - Falls back to "Possession" when labels missing

3. **xThreat**: Simplified distance-based calculation
   - Not a full xThreat model (would need pitch control, passing lanes, etc.)
   - Good enough for Phase 1.2, can improve in Phase 3

## Next Steps (Phase 2)

After Phase 1.2 completion:

1. **Validate Results**
   - Review outcome distributions in logs
   - Spot-check samples against video (if available)
   - Verify goal/shot rates are realistic (2-4% / 20-25%)

2. **Update Project Plan**
   - Mark Phase 1.2 complete in `notes/CORNER_GNN_PLAN.md`
   - Update README.md with new datasets

3. **Begin Phase 2: Graph Construction**
   - Extract player positions from tracking data
   - Build spatial graphs (nodes = players, edges = relationships)
   - Implement adjacency matrix strategies
   - Prepare graph dataset for GNN training

## Technical Details

### Coordinate Systems

**StatsBomb**: 120 x 80 units
- Goal at (120, 40)
- Corners at (120, 0) or (120, 80)

**SkillCorner**: Meters, center origin
- Goal at (+52.5m, 0) (for 105m x 68m pitch)
- Need to transform to standard coordinates

**SoccerNet**: Varies by match
- Use gameinfo.ini for pitch dimensions

### Data Flow

```
Phase 1.1 Datasets
     ↓
Outcome Labeling (Phase 1.2)
     ├─ StatsBomb corners → label_statsbomb_outcomes_v2.py
     ├─ SkillCorner corners → label_skillcorner_outcomes.py
     └─ SoccerNet corners → label_soccernet_outcomes.py
     ↓
Outcome-Labeled Datasets
     ↓
integrate_corner_datasets.py
     ↓
Unified Dataset (with outcomes)
     ↓
Ready for Phase 2 (Graph Construction)
```

## Troubleshooting

### If job fails:

1. **Check logs**: `logs/phase1_2_outcomes_<job_id>.err`
2. **Library issues**: Verify libstdc++ fix is applied
3. **API errors**: StatsBomb may rate-limit, add delays if needed
4. **File not found**: Ensure Phase 1.1 completed successfully

### If outcomes look wrong:

1. **Run test script**: `sbatch scripts/slurm/test_outcome_detection.sh`
2. **Check first 10 matches**: Review detailed output
3. **Adjust time window**: Change `max_time_window` if needed
4. **Manual validation**: Compare with video clips

## References

- Phase 1.1 Commit: `fc55ba2`
- GNN Plan: `notes/CORNER_GNN_PLAN.md`
- CLAUDE.md: Development guidelines
- README.md: Project documentation

---

**Status**: Ready for execution ✅
**Estimated Completion**: ~6 hours after job submission
**Next Phase**: Graph Construction (Phase 2)
