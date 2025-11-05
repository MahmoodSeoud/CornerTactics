# XGBoost Feature Importance Visualization

This directory contains scripts for creating TacticAI-style visualizations of XGBoost feature importance with soccer pitch context.

## Overview

After training XGBoost baselines, you can generate visually engaging feature importance plots that overlay the most important features on a soccer pitch, similar to TacticAI's tactical analysis style.

## Workflow

### 1. Train XGBoost Model

First, train the XGBoost outcome baseline model:

```bash
# Submit SLURM job for training
sbatch scripts/slurm/train_outcome_baselines_v100.sh

# Or run locally (for testing)
python scripts/training/train_outcome_baselines.py \
    --models xgboost \
    --output-dir results/baselines
```

This automatically:
- Trains the XGBoost model
- Extracts feature importance
- Saves to `results/baselines/feature_importance.json`
- Creates basic bar chart `results/baselines/feature_importance_basic.png`

### 2. Create Pitch-Based Visualizations

Generate TacticAI-style pitch visualizations:

```bash
# Via SLURM (recommended)
sbatch scripts/slurm/visualize_feature_importance.sh

# Or run locally
python scripts/visualization/visualize_feature_importance.py \
    --importance-path results/baselines/feature_importance.json \
    --output-dir data/results/feature_importance
```

## Output Visualizations

The script creates 3 types of visualizations:

### 1. Feature Category Bar Chart
**File**: `feature_categories_bar.png`

Horizontal bar chart grouping features by category:
- **Spatial** (red): x, y, distance_to_goal, distance_to_ball
- **Kinematic** (teal): vx, vy, velocity_magnitude
- **Contextual** (yellow): angle_to_goal, team_flag, in_penalty_box
- **Density** (green): num_players_within_5m, local_density_score
- **Zonal** (purple): in_six_yard, near_post, far_post

### 2. Spatial Feature Heatmap
**File**: `spatial_heatmap.png`

Soccer pitch with color-coded zones showing spatial feature importance:
- Penalty area, 6-yard box, near post, far post, central zone
- Darker zones = higher importance
- White → Yellow → Orange → Red gradient

### 3. Top Features with Pitch Annotations (TacticAI Style)
**File**: `top_features_pitch.png`

Professional TacticAI-style visualization:
- **Arrows**: Show directional features (distance_to_goal, velocity)
- **Zones**: Highlight important areas (penalty box, 6-yard box)
- **Circles**: Represent density features
- Color intensity indicates feature importance

## Feature Categories

The visualization groups features into 5 categories:

| Category | Features |
|----------|----------|
| **Spatial** | x, y, distance_to_goal, distance_to_ball, avg_x, std_x, avg_y, std_y |
| **Kinematic** | vx, vy, velocity_magnitude, velocity_angle, avg_vel, max_vel |
| **Contextual** | angle_to_goal, angle_to_ball, team_flag, in_penalty_box, attacking_players, defending_players |
| **Density** | num_players_within_5m, local_density_score, avg_density, max_density, penalty_area_density, six_yard_density |
| **Zonal** | in_six_yard, near_post, far_post, central_zone |

## Customization

### Change Number of Top Features

```bash
python scripts/visualization/visualize_feature_importance.py \
    --importance-path results/baselines/feature_importance.json \
    --output-dir data/results/feature_importance \
    --top-n 15  # Show top 15 features (default: 10)
```

### Modify Feature Categories

Edit `FEATURE_CATEGORIES` dict in `visualize_feature_importance.py`:

```python
FEATURE_CATEGORIES = {
    'spatial': ['x', 'y', 'distance_to_goal', ...],
    'kinematic': ['vx', 'vy', ...],
    # Add custom categories here
}
```

### Customize Pitch Visuals

Edit `feature_visuals` dict in `plot_top_features_with_pitch_annotations()`:

```python
feature_visuals = {
    'distance_to_goal': {'pos': (115, 40), 'type': 'arrow', 'target': (120, 40)},
    'penalty': {'pos': (111, 40), 'type': 'zone', 'bounds': [102, 120, 18, 62]},
    # Add custom visual mappings here
}
```

## Example Output

### What People Usually Do

In soccer analytics research (TacticAI, StatsBomb, etc.), feature importance is typically visualized as:

1. **Bar charts** (most common): Simple horizontal bars showing feature importance
2. **Pitch heatmaps**: Overlay importance on pitch zones
3. **Tactical arrows**: Show directional features with arrows on pitch
4. **Combined views**: Multiple subplots showing different perspectives

Our implementation combines all these approaches!

### Inspiration from TacticAI

TacticAI (Google DeepMind) uses:
- Pitch overlays with colored zones
- Arrows showing player movements and passes
- Heatmaps for density and positioning
- Annotations with feature names and values

## Dependencies

```bash
pip install matplotlib mplsoccer numpy
```

## Troubleshooting

### Error: "No spatial features found"

If the spatial heatmap is skipped, your features may not match the expected naming:
- Check feature names in `feature_importance.json`
- Update `FEATURE_CATEGORIES` in the script

### Error: "Cannot load feature_importance.json"

Ensure you've run XGBoost training first:
```bash
python scripts/training/train_outcome_baselines.py --models xgboost
```

## References

- **TacticAI Paper**: [A Graph Neural Network Deep-Dive into Successful Counterattacks](https://arxiv.org/abs/2310.09943)
- **mplsoccer**: [Soccer pitch visualization library](https://mplsoccer.readthedocs.io/)
- **XGBoost**: [Feature importance documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score)
