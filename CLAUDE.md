# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a **data extraction project** for corner kick prediction research using StatsBomb's open event data and 360-degree freeze frame player positioning data.

**Purpose**: Extract and process corner kick data with player positions for binary shot prediction research

**Dataset**: 1,933 corners with 360-degree freeze frames from StatsBomb Open Data

## Development Environment

**Conda Environment**: `robo`
```bash
conda activate robo
```

**Python Version**: 3.x

## Project Structure

```
CornerTactics/
├── scripts/                                    # Data processing pipeline
│   ├── download_statsbomb_events.py           # Download StatsBomb event data
│   ├── download_statsbomb_360_freeze_frames.py # Download 360 freeze frames
│   ├── 01_extract_corners.py                  # Extract corners with freeze frames
│   ├── 02_extract_shot_labels.py              # Extract binary shot labels
│   ├── 03_extract_valid_features.py           # Extract 22 valid features (no leakage)
│   ├── 04_create_splits.py                    # Create train/val/test splits
│   └── 05_extract_raw_spatial.py              # Extract raw coordinates (ablation)
├── src/
│   └── statsbomb_loader.py                    # StatsBomb data loader class
├── data/                                       # Data directory (gitignored)
│   ├── statsbomb/                             # Raw downloaded data
│   └── processed/                             # Processed corner data
├── tests/                                      # Test suite
├── docs/
│   └── STATSBOMB_DATA_GUIDE.md               # Data documentation
├── requirements.txt
├── CLAUDE.md                                   # This file
└── README.md
```

## Data Pipeline

Run scripts in order:

```bash
conda activate robo

# 1. Download raw data
python scripts/download_statsbomb_events.py
python scripts/download_statsbomb_360_freeze_frames.py

# 2. Extract and process
python scripts/01_extract_corners.py           # → corners_with_freeze_frames.json
python scripts/02_extract_shot_labels.py       # → corners_with_shot_labels.json
python scripts/03_extract_valid_features.py    # → corners_features_temporal_valid.csv
python scripts/04_create_splits.py             # → train/val/test_indices.csv

# 3. Optional: raw coordinates for ablation
python scripts/05_extract_raw_spatial.py       # → corners_raw_spatial_features.csv
```

## Feature Engineering

### Valid Features (22 total, no temporal leakage)

**Best 4 Features (Configuration A):**
- `corner_y` - Corner y-coordinate
- `defending_to_goal_dist` - Mean defender distance to goal
- `defending_near_goal` - Defenders in 6-yard box
- `defending_depth` - Defensive line spread (std Y)

**Additional 18 Features (Configuration B):**

Event Metadata:
- `minute`, `second`, `period`, `corner_x`
- `attacking_team_goals`, `defending_team_goals`

Freeze-Frame:
- `total_attacking`, `total_defending`
- `attacking_in_box`, `defending_in_box`
- `attacking_near_goal`
- `attacking_density`, `defending_density`
- `numerical_advantage`, `attacker_defender_ratio`
- `attacking_to_goal_dist`, `keeper_distance_to_goal`
- `corner_side`

### Excluded Features (13, temporal leakage)

These features encode information only available AFTER the corner kick:
- `is_shot_assist` - directly encodes target
- `pass_end_x`, `pass_end_y` - actual ball landing position
- `pass_length`, `pass_angle` - computed from actual trajectory
- `duration`, `has_recipient`, `pass_recipient_id`
- `has_pass_outcome`, `pass_outcome`, `pass_outcome_encoded`
- `is_aerial_won`, `is_cross_field_switch`

## Dataset Statistics

```
Corners with 360 Data:  1,933
Shot Rate:              29.0% (560 shots)
No-Shot Rate:           71.0% (1,373)
```

**Train/Val/Test Split** (match-based, no overlap):
- Train: 1,155 (60%)
- Val: 371 (19%)
- Test: 407 (21%)

## Code Philosophy

- Straightforward, data-oriented code
- Efficient pandas operations
- Clear variable names
- Think like John Carmack: fix problems, don't work around them

## Important Notes

1. **Temporal Validity**: Only use features available at t=0 (corner kick moment)
2. **Match-Based Splits**: Prevents data leakage from same-match corners
3. **StatsBomb Coordinates**: 120x80 pitch (x: 0-120, y: 0-80)
4. **Data Directory**: All data in `data/` (gitignored, ~11GB)

## Git Workflow

- Never commit data files (gitignored)
- Keep commit messages concise
