# CornerTactics: Paper Methods and Results

**Updated:** November 25, 2025
**Status:** Comprehensive document for paper writing

---

## EXECUTIVE SUMMARY

This document contains all methods, experiments, and results needed to write a comprehensive paper on corner kick outcome prediction using StatsBomb's open data with 360° freeze frames.

**Key Finding:** Corner kick outcomes are essentially unpredictable using only pre-kick information. All models perform at or below baseline.

---

## 1. DATASET

### Source
- **Provider**: StatsBomb Open Data (free, publicly available)
- **Data Types**: Event data + 360° freeze frame player positioning
- **Total Corners with 360° Data**: 1,933

### Class Distribution

#### Binary Shot Prediction
| Class | Count | Percentage |
|-------|-------|------------|
| No Shot | 1,373 | 71.0% |
| Shot | 560 | 29.0% |
| **Total** | **1,933** | 100% |

#### Multi-Class Outcome Prediction
| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| Ball Receipt | 1,050 | 54.3% | Attacking team receives ball |
| Clearance | 453 | 23.4% | Defending team clears |
| Goalkeeper | 196 | 10.1% | Keeper collects/punches |
| Other | 234 | 12.1% | Duel, foul, pressure, etc. |
| **Total** | **1,933** | 100% | |

### Coordinate System
- **Pitch dimensions**: 120 × 80 units
- **Goal center**: (120, 40)
- **Penalty box**: x > 102, 18 < y < 62
- **Corner positions**: (0, 0), (0, 80), (120, 0), (120, 80)

---

## 2. DATA SPLIT METHODOLOGY

### Match-Based Stratified Split
To prevent data leakage from corners in the same match appearing in different sets:

| Set | Samples | Percentage | Matches |
|-----|---------|------------|---------|
| Train | 1,155 | 59.8% | ~194 |
| Validation | 371 | 19.2% | ~62 |
| Test | 407 | 21.1% | ~67 |
| **Total** | **1,933** | 100% | 323 |

### Split Properties
- **Stratified by outcome**: Class distribution preserved across splits
- **No match overlap**: Zero matches shared between train/val/test
- **Random seed**: 42 (reproducible)

### Why Match-Based Splitting?
Multiple corners from the same match share:
- Same teams and players
- Similar tactical patterns
- Correlated outcomes

Random splitting would leak this information from train to test.

---

## 3. PREDICTION TASKS

### Task 1: Binary Shot Prediction
- **Question**: Did the corner kick lead to a shot by the attacking team?
- **Classes**: Shot (1) vs No Shot (0)
- **Baseline**: Always predict "No Shot" → 71.74% accuracy
- **Evaluation**: Accuracy, AUC-ROC

### Task 2: Multi-Class Outcome Prediction
- **Question**: What was the immediate outcome of the corner kick?
- **Classes**: Ball Receipt, Clearance, Goalkeeper, Other
- **Baseline**: Always predict "Ball Receipt" → 53.07% accuracy
- **Evaluation**: Accuracy, Macro F1, AUC-ROC (one-vs-rest)

---

## 4. SHOT LABELING METHODOLOGY

### Definition (Following TacticAI)
- **Lookahead window**: 5 events after corner kick
- **Team filter**: Only shots from attacking team (corner-taking team)
- **Shot outcomes included**: ALL types
  - Goal, Saved, Post, Off Target, Wayward, Blocked, Saved Off Target, Saved to Post

### Validation Against StatsBomb's is_shot_assist
```
                    leads_to_shot=0   leads_to_shot=1
is_shot_assist=0         1372              190
is_shot_assist=1            1              370
```
- **99.7% agreement** with StatsBomb direct assists (370/371)
- **190 additional indirect assists** captured (headers, scrambles)
- **1 disagreement**: Shot occurred after 5-event window

---

## 5. TEMPORAL DATA LEAKAGE ANALYSIS

### Critical Discovery
Many features that appeared predictive actually contained information only available AFTER the outcome occurred.

### Features Removed (Leaked)

| Feature | Why Leaked | Evidence |
|---------|------------|----------|
| `is_shot_assist` | Directly encodes target | MCC = 0.753 with target |
| `has_recipient` | Only known after pass completes | Requires outcome |
| `pass_end_x/y` | Actual landing location | Matches next event location exactly |
| `pass_length` | Computed from actual landing | Derived from leaked data |
| `pass_angle` | Computed from actual landing | Derived from leaked data |
| `duration` | Time until next event | Only known after event |
| `is_cross_field_switch` | Post-hoc categorization | 99.6% outcome-correlated |
| `is_inswinging/outswinging` | May encode outcome | Ambiguous timing |

### Evidence: pass_end_x/y is Actual, Not Intended
```python
# Corner event from StatsBomb JSON:
Corner location: [120.0, 80.0]
Pass end_location: [114.5, 75.0]

# Next event (Ball Receipt):
Location: [114.5, 75.0]  # EXACT MATCH with pass_end
```
This proves `pass_end` is where the ball actually landed, not where the kicker intended.

---

## 6. VALID FEATURE SET (22 Features)

Only features available at t=0 (moment of corner kick execution):

### Event Data (7 features)
| Feature | Description | Type |
|---------|-------------|------|
| `minute` | Match minute | Continuous |
| `second` | Second within minute | Continuous |
| `period` | First (1) or second (2) half | Categorical |
| `corner_x` | Corner position X | Continuous |
| `corner_y` | Corner position Y | Continuous |
| `attacking_team_goals` | Attacking team's current score | Discrete |
| `defending_team_goals` | Defending team's current score | Discrete |

### 360° Freeze Frame Data (15 features)
Captured at the exact moment of corner kick:

**Player Counts**
| Feature | Description |
|---------|-------------|
| `total_attacking` | Total attacking players in frame |
| `total_defending` | Total defending players in frame |
| `attacking_in_box` | Attackers inside penalty area |
| `defending_in_box` | Defenders inside penalty area |
| `attacking_near_goal` | Attackers within 6 yards of goal |
| `defending_near_goal` | Defenders within 6 yards of goal |

**Spatial Metrics**
| Feature | Description |
|---------|-------------|
| `attacking_density` | Spatial concentration of attackers |
| `defending_density` | Spatial concentration of defenders |
| `numerical_advantage` | Attackers minus defenders |
| `attacker_defender_ratio` | Ratio of attackers to defenders |

**Positional Distances**
| Feature | Description |
|---------|-------------|
| `defending_depth` | Average Y position of defensive line |
| `attacking_to_goal_dist` | Average attacker distance to goal center |
| `defending_to_goal_dist` | Average defender distance to goal center |
| `keeper_distance_to_goal` | Goalkeeper distance to goal center |

**Other**
| Feature | Description |
|---------|-------------|
| `corner_side` | Left (0) or right (1) corner |

---

## 7. MODELS

### Model Architectures

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
```

#### XGBoost
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<class_ratio>,  # For binary
    random_state=42
)
```

#### MLP (Neural Network)
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),      # Binary
    hidden_layer_sizes=(128, 64, 32), # Multi-class
    max_iter=1000,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

### Preprocessing
- **Scaling**: StandardScaler for MLP and Random Forest
- **No scaling**: XGBoost (tree-based, scale-invariant)
- **Class imbalance**: Handled via `class_weight='balanced'` or `scale_pos_weight`

---

## 8. RESULTS: BINARY SHOT PREDICTION

### Performance Metrics

| Model | Train Acc | Val Acc | Test Acc | Val AUC | Test AUC |
|-------|-----------|---------|----------|---------|----------|
| Random Forest | 92.29% | 66.85% | 59.95% | 0.5497 | 0.4526 |
| XGBoost | 99.39% | 63.61% | 60.44% | 0.5150 | 0.5095 |
| MLP | 70.48% | 72.78% | 70.52% | 0.5889 | 0.4324 |
| **Baseline** | - | - | **71.74%** | - | 0.5000 |

### Key Observations
1. **All models at or below baseline**: Best test accuracy (70.52%) < baseline (71.74%)
2. **No predictive power**: Test AUC values 0.43-0.51 (random = 0.50)
3. **Severe overfitting**: Train accuracy 70-99%, test accuracy 60-71%
4. **MLP least overfit**: Smallest train-test gap

### Interpretation
The models have **zero predictive power** for shot prediction. The AUC values below 0.50 indicate the models are actually worse than random guessing on the test set.

---

## 9. RESULTS: MULTI-CLASS OUTCOME PREDICTION

### Performance Metrics

| Model | Train Acc | Val Acc | Test Acc | Val F1 | Test F1 | Test AUC |
|-------|-----------|---------|----------|--------|---------|----------|
| Random Forest | 88.48% | 37.74% | 43.00% | 0.2624 | 0.2819 | 0.5219 |
| XGBoost | 98.53% | 49.06% | 49.39% | 0.2327 | 0.2237 | 0.5291 |
| MLP | 57.49% | 53.37% | 50.86% | 0.1979 | 0.1734 | 0.4995 |
| **Baseline** | - | - | **53.07%** | - | - | - |

### Per-Class Performance (Best Model: MLP)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ball Receipt | 0.52 | 0.95 | 0.68 | 216 |
| Clearance | 0.08 | 0.01 | 0.02 | 97 |
| Goalkeeper | 0.00 | 0.00 | 0.00 | 44 |
| Other | 0.00 | 0.00 | 0.00 | 50 |
| **Macro Avg** | 0.15 | 0.24 | 0.17 | 407 |

### Key Observations
1. **All models at or below baseline**: Best accuracy (50.86%) < baseline (53.07%)
2. **Models predict majority class only**: 95% recall for Ball Receipt, ~0% for others
3. **Poor F1 scores**: Macro F1 of 0.17-0.28 indicates failure across classes
4. **Minority classes ignored**: Goalkeeper and Other never predicted correctly

### Interpretation
The models have learned nothing useful. They simply predict "Ball Receipt" for almost every sample, achieving roughly the majority class baseline.

---

## 10. KEY FINDINGS

### Finding 1: Corner Outcomes Are Unpredictable
- **Binary prediction**: AUC ~0.45-0.51 (worse than random)
- **Multi-class prediction**: Accuracy below majority baseline
- **Conclusion**: Pre-kick positioning provides no predictive signal

### Finding 2: Execution Dominates Setup
The lack of predictive power suggests:
- How the corner is delivered matters more than player positioning
- Small variations in kick trajectory create large outcome differences
- Player skill and split-second decisions dominate outcomes

### Finding 3: Previous High Performance Was Leakage
- **Original results**: 87.97% accuracy (with leaked features)
- **Valid results**: 70.52% accuracy (no leakage)
- **Leakage contribution**: ~17% of apparent accuracy was artificial

### Finding 4: Freeze Frame Data Has Limited Value
Despite having player positions at the moment of kick:
- 15 freeze frame features contribute minimally
- Spatial configuration doesn't predict outcomes
- Dynamic factors (movement, timing) may be more important

---

## 11. LIMITATIONS

1. **Static snapshots**: Freeze frames capture one moment, not player movement
2. **Missing attributes**: No player height, heading ability, historical performance
3. **Sample size**: 1,933 corners limits statistical power
4. **Single dataset**: Results may not generalize to other leagues/competitions
5. **Feature engineering**: Other spatial features might be more predictive
6. **Inherent randomness**: Corner outcomes may be fundamentally stochastic

---

## 12. PAPER WRITING GUIDANCE

### Abstract (150 words)
"We investigate the predictability of corner kick outcomes using StatsBomb's open event data with 360-degree freeze frame player positioning. After carefully removing temporally leaked features that contained outcome information, we train Random Forest, XGBoost, and MLP classifiers on 1,933 corners using 22 valid features. For binary shot prediction, all models achieve accuracy at or below the 71.74% baseline (always predict 'no shot'), with AUC values of 0.43-0.51 indicating no predictive power. For multi-class outcome prediction (Ball Receipt, Clearance, Goalkeeper, Other), models achieve 43-51% accuracy versus a 53% baseline, essentially learning only to predict the majority class. These results suggest that corner kick outcomes are fundamentally unpredictable from pre-kick spatial configuration alone, and that execution quality, player skill, and in-the-moment decisions dominate outcomes over static positioning."

### Methods Section Key Points
1. **Data source**: StatsBomb Open Data, 1,933 corners with 360° freeze frames
2. **Split**: 60/20/20 match-based stratified to prevent leakage
3. **Features**: 22 temporally valid features (7 event, 15 freeze frame)
4. **Leakage removal**: 8+ features removed including pass_end, is_shot_assist, duration
5. **Tasks**: Binary shot prediction + 4-class outcome prediction
6. **Models**: Random Forest, XGBoost, MLP with class balancing

### Results Section Key Points
1. **Binary**: All models ≤ baseline (71.74%), AUC ~0.5
2. **Multi-class**: All models ≤ baseline (53.07%), F1 ~0.2
3. **Overfitting**: High train accuracy, poor generalization
4. **Majority class**: Models learn only to predict dominant class

### Discussion Points
1. **Negative result is valuable**: Knowing corners are unpredictable is useful
2. **Implications for practice**: Focus on execution over positioning
3. **Future work**: Video analysis, trajectory prediction, player tracking
4. **Comparison to literature**: Aligns with inherent randomness in set pieces

---

## 13. SUMMARY TABLE

| Metric | Value |
|--------|-------|
| Total corners | 1,933 |
| Train/Val/Test split | 60/20/20 (1155/371/407) |
| Valid features | 22 |
| Leaked features removed | 8+ |
| **Binary Shot Prediction** | |
| Shot rate | 29.0% |
| Best model | MLP |
| Best test accuracy | 70.52% |
| Baseline accuracy | 71.74% |
| Best test AUC | 0.5095 (XGBoost) |
| **Multi-Class Outcome** | |
| Classes | 4 (Ball Receipt, Clearance, Goalkeeper, Other) |
| Best model | MLP |
| Best test accuracy | 50.86% |
| Baseline accuracy | 53.07% |
| Best test F1 (macro) | 0.2819 (Random Forest) |
| **Conclusion** | **No predictive power** |

---

## 14. FILES AND REPRODUCIBILITY

### Data Files
```
data/processed/
├── corners_features_temporal_valid.csv  # Clean dataset (1933 × 26)
├── temporal_valid_features.json         # Feature metadata
├── train_indices.csv                    # Training indices
├── val_indices.csv                      # Validation indices
└── test_indices.csv                     # Test indices
```

### Scripts
```
scripts/
├── 14_extract_temporally_valid_features.py  # Extract clean features
└── 15_retrain_without_leakage.py            # Train both tasks
```

### Results
```
results/no_leakage/
└── training_results.json  # Full metrics for all models
```

### Running the Experiments
```bash
# Extract features (if needed)
python scripts/14_extract_temporally_valid_features.py

# Train models and get results
python scripts/15_retrain_without_leakage.py
```

---

*This document contains all information needed to write a comprehensive paper on corner kick outcome prediction.*

*Last updated: November 25, 2025*
