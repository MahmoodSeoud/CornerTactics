# Raw Data Locations Guide

## Overview
This document provides the definitive guide to all raw and processed data locations for the CornerTactics project, specifically for training models on the 34K corner kick dataset.

**Last Updated**: November 2025
**Primary Dataset**: 34,049 corner kicks from StatsBomb Open Data

---

## 1. Primary Raw Dataset (34K Corners)

### 1.1 Corner Sequences with Event Data
**File**: `data/analysis/corner_sequences_detailed.json`
- **Size**: ~405 KB
- **Records**: 34,049 corner kicks (100 loaded in memory for testing)
- **Format**: JSON array of corner sequences
- **Contents**:
  - Corner event details (location, pass parameters, technique)
  - Next 15 events after each corner
  - Player IDs and positions
  - Timestamps and durations
  - Event types and outcomes

**Sample Structure**:
```json
{
  "corner_event": {
    "id": "corner_id",
    "timestamp": "00:13:13.843",
    "location": [120.0, 0.0],
    "player": {"id": 8804, "name": "Jonas Hofmann"},
    "pass": {
      "end_location": [113.9, 47.6],
      "length": 47.8,
      "angle": 1.699,
      "inswinging": true,
      "technique": {"id": 104, "name": "Inswinging"}
    }
  },
  "next_events": [
    {"type": {"name": "Ball Receipt"}, "player": {...}, "location": [...]},
    // ... up to 15 events
  ],
  "corner_features": {
    "possession_team": {"id": 904, "name": "Bayer Leverkusen"},
    "play_pattern": {"name": "From Corner"},
    // ... additional metadata
  }
}
```

### 1.2 Transition Probability Matrix
**File**: `data/analysis/corner_transition_matrix.csv`
- **Size**: 40 KB
- **Dimensions**: 1 × 576 event types
- **Format**: CSV with probabilities
- **Contents**: P(next_event | corner) for all possible events

**Top Event Types (for classification)**:
1. Ball Receipt (57.8%)
2. Clearance (22.9%)
3. Goal Keeper (7.6%)
4. Duel (Aerial Lost) (3.4%)
5. Pressure (2.9%)
6. Foul Committed (1.3%)
7. Ball Recovery (1.0%)
8. Block (0.6%)
9. Pass (Goal Kick) (0.2%)
10. Substitution (0.1%)
... plus 5+ more rare events

### 1.3 Summary Report
**File**: `data/analysis/corner_transition_report.md`
- **Size**: 1.5 KB
- **Format**: Markdown report
- **Contents**: Statistical summary, temporal analysis, outcome rates

---

## 2. Original StatsBomb 360 Data (For Reference)

### 2.1 Single-Frame Corner Data
**File**: `data/raw/statsbomb/corners_360.csv`
- **Size**: ~18.9 MB
- **Records**: 1,118 corners with 360 freeze frames
- **Format**: CSV with JSON freeze frame data
- **Note**: Subset of the 34K dataset with full player positioning

### 2.2 Processed Features
**Directory**: `data/features/`
```
data/features/
├── node_features/
│   ├── statsbomb_player_features.parquet
│   └── statsbomb_player_features.csv
└── temporal/
    ├── skillcorner_temporal_features.parquet
    └── skillcorner_temporal_features.csv
```

---

## 3. Graph Datasets (Not Used for Raw Training)

**Directory**: `data/graphs/adjacency_team/`
- `statsbomb_graphs.pkl` - Original single-frame graphs
- `statsbomb_temporal_augmented_with_receiver.pkl` - 5,814 augmented graphs
- **Note**: These contain engineered features, not used for raw baseline

---

## 4. Raw Features Available in StatsBomb Data

Based on the JSON structure, the following **raw features** are available by default:

### 4.1 Corner Event Features
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `timestamp` | string | Event timestamp | "00:13:13.843" |
| `minute` | int | Minute of match | 13 |
| `second` | int | Second within minute | 13 |
| `period` | int | Match period (1st/2nd half) | 1 |
| `location_x` | float | Corner x-coordinate | 120.0 |
| `location_y` | float | Corner y-coordinate | 0.0 |
| `end_location_x` | float | Target x-coordinate | 113.9 |
| `end_location_y` | float | Target y-coordinate | 47.6 |
| `pass_length` | float | Pass distance | 47.8 |
| `pass_angle` | float | Pass angle (radians) | 1.699 |
| `pass_height_id` | int | Height type ID | 3 |
| `inswinging` | bool | Inswinging corner | true |
| `switch` | bool | Crosses field | true |
| `technique_id` | int | Technique code | 104 |
| `body_part_id` | int | Body part code | 40 |

### 4.2 Contextual Features
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `possession_team_id` | int | Team with possession | 904 |
| `team_id` | int | Corner taking team | 904 |
| `player_id` | int | Corner taker ID | 8804 |
| `position_id` | int | Player position | 18 |
| `play_pattern_id` | int | Play pattern code | 2 |
| `possession` | int | Possession number | 26 |
| `duration` | float | Event duration | 2.213 |
| `related_events_count` | int | Related events | 1 |

### 4.3 Player Position Features (if freeze frame available)
| Feature | Type | Description |
|---------|------|-------------|
| `player_positions` | JSON | Array of player x,y coordinates |
| `player_teams` | JSON | Team affiliation for each player |
| `player_ids` | JSON | Player IDs in freeze frame |
| `keeper_position` | JSON | Goalkeeper location |

---

## 5. How to Load the Data

### 5.1 Load Corner Sequences (34K dataset)
```python
import json
import pandas as pd

# Load full 34K dataset
with open('data/analysis/corner_sequences_detailed.json', 'r') as f:
    corners = json.load(f)

print(f"Total corners: {len(corners)}")  # Should be 34,049
```

### 5.2 Load Transition Matrix
```python
# Load transition probabilities
transition_matrix = pd.read_csv('data/analysis/corner_transition_matrix.csv', index_col=0)
corner_probs = transition_matrix.loc['Corner']

# Get top 15 event types for classification
top_15_events = corner_probs.nlargest(15)
print(top_15_events)
```

### 5.3 Extract Raw Features for Training
```python
def extract_raw_features(corner):
    """Extract only raw StatsBomb features, no engineering."""
    event = corner['corner_event']
    features = corner.get('corner_features', {})

    # Basic features
    raw = {
        'minute': event.get('minute', 0),
        'second': event.get('second', 0),
        'period': event.get('period', 1),
        'location_x': event.get('location', [0, 0])[0],
        'location_y': event.get('location', [0, 0])[1],
    }

    # Pass features
    pass_data = event.get('pass', {})
    raw['end_location_x'] = pass_data.get('end_location', [0, 0])[0]
    raw['end_location_y'] = pass_data.get('end_location', [0, 0])[1]
    raw['pass_length'] = pass_data.get('length', 0)
    raw['pass_angle'] = pass_data.get('angle', 0)
    raw['inswinging'] = int(pass_data.get('inswinging', False))
    raw['switch'] = int(pass_data.get('switch', False))

    # IDs
    raw['team_id'] = features.get('team.id', 0)
    raw['player_id'] = features.get('player.id', 0)
    raw['position_id'] = features.get('position.id', 0)
    raw['possession'] = features.get('possession', 0)

    return raw
```

---

## 6. Target Labels for Prediction

### 6.1 Task 1: Receiver Prediction
**Target**: First player to receive the ball
```python
def get_receiver(corner):
    """Extract receiver player ID from next events."""
    next_events = corner.get('next_events', [])
    for event in next_events:
        if 'Ball Receipt' in event.get('type', {}).get('name', ''):
            return event.get('player', {}).get('id')
    return None
```

### 6.2 Task 2: Event Outcome (15-class)
**Target**: Which of the 15 most common events occurs next
```python
def get_next_event_type(corner):
    """Extract the immediate next event type."""
    next_events = corner.get('next_events', [])
    if next_events:
        return next_events[0].get('type', {}).get('name', 'Unknown')
    return 'Unknown'
```

---

## 7. Directory Structure for Training

```
CornerTactics/
├── data/
│   ├── analysis/                     # Raw 34K dataset
│   │   ├── corner_sequences_detailed.json
│   │   ├── corner_transition_matrix.csv
│   │   └── corner_transition_report.md
│   │
│   └── splits/                       # Train/val/test splits (to be created)
│       ├── train_indices.json
│       ├── val_indices.json
│       └── test_indices.json
│
├── models/                           # Trained models (to be created)
│   ├── final/
│   │   ├── mlp_receiver_best.pth
│   │   ├── mlp_outcome_best.pth
│   │   ├── xgboost_receiver_best.pkl
│   │   └── xgboost_outcome_best.pkl
│   │
│   └── hyperparameter_search/
│       ├── mlp_search_results.json
│       └── xgboost_search_results.json
│
└── results/                          # All CSV/JSON outputs (to be created)
    ├── task1_receiver_prediction.csv
    ├── task2_outcome_classification.csv
    ├── task2_per_class_f1.csv
    ├── learning_curves_mlp.csv
    ├── learning_curves_xgboost.csv
    ├── feature_importance.csv
    ├── ablation_results.csv
    └── ... (all other required CSVs/JSONs)
```

---

## 8. Important Notes

1. **Use Raw Features Only**: For baseline training, use only the raw features provided by StatsBomb. Do NOT use engineered features from `data/features/` or `data/graphs/`.

2. **34K Dataset**: The primary dataset is in `corner_sequences_detailed.json` with 34,049 corners, not the 1,118 corners in `corners_360.csv`.

3. **Event Types**: Focus on the top 15 event types from the transition matrix for classification. The full 576 event types would be too sparse.

4. **Missing Data**: Some corners may not have all features. Handle missing values appropriately (imputation or masking).

5. **Class Imbalance**: The event outcomes are highly imbalanced (Ball Receipt: 57.8%, rare events: <1%). Use class weights or sampling strategies.

---

## 9. Quick Start Commands

```bash
# Navigate to project
cd /home/mseo/CornerTactics

# Activate environment
conda activate robo

# Load GCC for compatibility
module load GCC/13.3.0

# Start training script (to be created)
python scripts/train_raw_baseline.py
```

---

**Author**: CornerTactics Project
**Date**: November 2025
**Version**: 2.0 (Updated for 34K raw dataset)