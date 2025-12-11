# Corner Kick Data Extraction Pipeline

## Overview

Extract corner kick data from StatsBomb Open Data for binary shot prediction research.

**Dataset**: 1,933 corners with 360-degree freeze frames

## Pipeline

### Step 1: Download Raw Data

```bash
python scripts/download_statsbomb_events.py
python scripts/download_statsbomb_360_freeze_frames.py
```

Downloads:
- Match events from 3,464 matches
- 360 freeze frames for 323 matches with tracking data

### Step 2: Extract Corners

```bash
python scripts/01_extract_corners.py
```

Output: `data/processed/corners_with_freeze_frames.json`
- 1,933 corners with complete freeze frame data

### Step 3: Extract Shot Labels

```bash
python scripts/02_extract_shot_labels.py
```

Following TacticAI methodology:
- Look ahead 5 events after corner
- Check for threatening shots by attacking team
- Binary label: 1 (shot) or 0 (no shot)

Output: `data/processed/corners_with_shot_labels.json`
- Shot: 560 (29.0%)
- No Shot: 1,373 (71.0%)

### Step 4: Extract Valid Features

```bash
python scripts/03_extract_valid_features.py
```

Extracts 22 temporally valid features (no leakage):
- Event metadata (7): minute, second, period, corner_x, corner_y, attacking_team_goals, defending_team_goals
- Freeze-frame (15): player counts, densities, distances, positions

Excludes 13 leaked features: pass_end_x/y, is_shot_assist, etc.

Output: `data/processed/corners_features_temporal_valid.csv`

### Step 5: Create Splits

```bash
python scripts/04_create_splits.py
```

Match-based stratified split (60/20/20):
- Train: 1,155 samples (194 matches)
- Val: 371 samples (62 matches)
- Test: 407 samples (67 matches)

Output: `data/processed/{train,val,test}_indices.csv`

### Step 6 (Optional): Raw Spatial Features

```bash
python scripts/05_extract_raw_spatial.py
```

For ablation studies:
- Raw player coordinates (padded, sorted by distance to goal)
- Pairwise distances (marking structure)
- Spatial structure features

Output: `data/processed/corners_raw_spatial_features.csv`

## Feature Reference

### Valid Features (22)

**Configuration A (4 best):**
- corner_y
- defending_to_goal_dist
- defending_near_goal
- defending_depth

**Configuration B (18 additional):**
- minute, second, period, corner_x
- attacking_team_goals, defending_team_goals
- total_attacking, total_defending
- attacking_in_box, defending_in_box
- attacking_near_goal
- attacking_density, defending_density
- numerical_advantage, attacker_defender_ratio
- attacking_to_goal_dist, keeper_distance_to_goal
- corner_side

### Excluded Features (13, temporal leakage)

- is_shot_assist (target leakage)
- pass_end_x, pass_end_y (actual landing)
- pass_length, pass_angle (computed from actual)
- duration, has_recipient, pass_recipient_id
- has_pass_outcome, pass_outcome, pass_outcome_encoded
- is_aerial_won, is_cross_field_switch
