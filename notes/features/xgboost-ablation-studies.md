# XGBoost Ablation Studies

## Objective
Conduct three ablation experiments on XGBoost baseline for corner kick outcome prediction to understand:
1. Impact of temporal augmentation (t=0s only vs 5 temporal frames)
2. Impact of feature selection (all 22 players vs 5 closest to ball)
3. Feature importance analysis (which features matter most)

## Dataset Specifications
- Source: StatsBomb Open Data
- Base corners: 1,118 unique corners
- Augmented dataset: 5,814 graphs (5.2× augmentation)
- Temporal frames: t ∈ {-2s, -1s, 0s, +1s, +2s}
- Splits: 70% train / 15% val / 15% test (stratified by corner ID)
- Dataset file: `data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl`

## Tasks
1. **Receiver Prediction** (Task 1): 22-class classification
2. **Outcome Classification** (Task 2): 3-class (Shot/Clearance/Possession)

## Current Baseline Performance (from existing runs)
- Receiver: 68.87% Top-3 accuracy
- Outcome: 50.1% accuracy, Macro F1=0.419
  - Shot F1: 0.207 (18.2% class prevalence)
  - Clearance F1: 0.604 (52.0%)
  - Possession F1: 0.445 (29.9%)

## Feature Engineering
- Per-player features: 308 dimensions (22 players × 14 features)
  - 14 features per player: x, y, vx, vy, distance_to_goal, distance_to_ball, angle_to_goal, angle_to_ball, team_flag, in_penalty_box, velocity_magnitude, velocity_angle, num_players_within_5m, local_density_score
- Graph-level features: 29 dimensions (aggregated statistics)

## Implementation Notes

### Experiment 1: Temporal Augmentation
- Filter to t=0s frames only (~1,118 corners)
- Train XGBoost on both configurations
- Compare performance metrics
- Expected: Small improvement from augmentation (~1-2pp)

### Experiment 2: Feature Selection
- Extract features for 5 closest players to ball landing zone
- Reduced features: 70 dimensions (5 × 14)
- For outcome task: aggregate to graph-level (~9 features)
- Expected: Performance degradation (~3-5pp) due to information loss

### Experiment 3: Feature Importance
- Use XGBoost's gain metric
- Visualize top 15 features
- Categorize by type (distance, position, density, formation, angular)
- Expected: Distance and density features to dominate

## Key Implementation Details

### Filtering t=0s frames
Corner IDs contain temporal markers:
- Format: `match_123_event_456_t0` or `match_123_event_456_t+0.0`
- Filter logic: Check for `_t0` or `_t+0` or `_t+0.0` in corner_id

### 5 Closest Players Selection
- Use receiver_location as ball landing zone proxy
- Calculate Euclidean distances from all players
- Select 5 with minimum distance
- Preserve original 14-dimensional feature vectors

### XGBoost Hyperparameters (from baseline)
```python
receiver_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'num_class': 22,
    'eval_metric': 'mlogloss',
    'seed': 42
}

outcome_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'seed': 42
}
```

## Deliverables
1. Python script: `scripts/analysis/xgboost_ablation_experiments.py`
2. SLURM job script: `scripts/slurm/xgboost_ablation.sh`
3. Results directory: `results/ablation_studies/`
4. LaTeX tables for paper (3 experiments)
5. Feature importance visualization
6. Interpretation summaries

## File Structure
```
results/ablation_studies/
├── exp1_temporal_augmentation/
│   ├── results.json
│   ├── latex_table.txt
│   └── interpretation.md
├── exp2_feature_selection/
│   ├── results.json
│   ├── latex_table.txt
│   └── interpretation.md
└── exp3_feature_importance/
    ├── feature_importance.json
    ├── feature_importance_plot.png
    ├── latex_enumeration.txt
    └── interpretation.md
```

## Testing Strategy
1. Unit tests for filtering logic (t=0s extraction)
2. Unit tests for 5-closest-players selection
3. Integration test for full pipeline
4. Validation checks on result consistency

## Questions & Answers
- Q: Should we use validation or test set for final metrics?
  A: Test set for final reporting (following TacticAI methodology)

- Q: How to handle missing receiver_location?
  A: Use center of penalty box [102.0, 40.0] as fallback (StatsBomb coordinates)

- Q: Should we run receiver prediction or just outcome classification?
  A: Both tasks to be comprehensive, but outcome is primary focus

## Progress Log
- Created feature branch: `feature/xgboost-ablation-studies`
- Reviewed existing baseline implementation
- Documented dataset specifications and experiment designs
