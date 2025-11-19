# CornerTactics: Paper Methods and Results Summary

**For LLM Comprehension - Updated November 19, 2025**

---

## OUTDATED FILES TO IGNORE

The following files contain outdated information and should NOT be used:

### Completely Outdated (Different Results)
- `docs/METHODS_AND_RESULTS_SUMMARY.md` - Old results with leaked features
- `docs/DATA_LEAKAGE_FINDINGS.md` - Old numbers (422 shots, not 560)
- `docs/FEATURE_IMPORTANCE_RESULTS.md` - Old results (78.8% not 68.2%)
- `ABLATION_RESULTS.md` - All results used leaked features (86% accuracy is INVALID)
- `ABLATION_STUDY_ANALYSIS_PROMPT.md` - Based on leaked results
- `INDIVIDUAL_ABLATION_FINDINGS.md` - Based on leaked results
- `COMPLETE_ABLATION_ANALYSIS_PROMPT.md` - Based on leaked results
- `results/ablation/*` - All ablation results invalid
- `results/evaluation_report.md` - Old 4-class results

### Partially Outdated
- `notes/features/task-7-shot-label-extraction.md` - Shot definition updated
- `notes/features/task-8-binary-shot-models.md` - Results outdated

### CURRENT/VALID Files
- `docs/CLEAN_BASELINE_RESULTS.md` - **USE THIS** for model performance
- `docs/SHOT_LABEL_VERIFICATION.md` - **USE THIS** for label methodology
- `notes/features/data-leakage-analysis.md` - **USE THIS** for complete analysis

---

## 1. DATASET

### Source
- **Provider**: StatsBomb Open Data
- **Total Corners with 360° Data**: 1,933
- **Class Distribution**: 560 shots (29.0%), 1,373 no shots (71.0%)
- **Imbalance Ratio**: 2.45:1

### Coordinate System
- **Pitch**: 120 × 80 units
- **Goal Center**: (120, 40)
- **Penalty Box**: x > 102, 18 < y < 62

---

## 2. SHOT LABEL DEFINITION

### Methodology (Based on TacticAI)
- **Lookahead Window**: 5 events after corner kick
- **Team Filter**: Only shots from attacking team (corner-taking team)
- **Shot Outcomes Included**: ALL shot types
  - Goal
  - Saved
  - Post
  - Off Target
  - Wayward
  - **Blocked** (included)
  - Saved Off Target
  - Saved to Post

### Validation Against StatsBomb's is_shot_assist
```
                    leads_to_shot=0   leads_to_shot=1
is_shot_assist=0         1372              190
is_shot_assist=1            1              370
```

- **99.7% agreement** with StatsBomb direct assists (370/371)
- **190 additional indirect assists** captured (headers, scrambles)
- **1 disagreement**: shot occurred after 5-event window

---

## 3. TEMPORAL DATA LEAKAGE ANALYSIS

### Critical Discovery
**7 features removed** because they contain information only available AFTER the outcome:

| Feature | MCC | Why Leaked |
|---------|-----|------------|
| `is_shot_assist` | 0.753 | Directly encodes if next event is shot |
| `has_recipient` | 0.303 | Only known after pass completes |
| `pass_end_x` | -0.143 | Actual landing location (not intended) |
| `pass_end_y` | - | Actual landing location (not intended) |
| `pass_length` | 0.146 | Computed from actual landing |
| `pass_angle` | -0.083 | Computed from actual landing |
| `duration` | 0.157 | Time until next event |

### Evidence for pass_end_x/y Leakage
Analyzed StatsBomb JSON events:
```python
# Corner with recipient:
Corner location: [120.0, 80.0]
Pass end_location: [114.5, 75.0]
Next event (Ball Receipt): [114.5, 75.0]  # EXACT MATCH
```
This proves `pass_end` is the **actual landing location**, not intended target.

---

## 4. CLEAN FEATURE SET (36 Features)

### Freeze-Frame Derived (18 features)
**Player Counts**:
- `attacking_in_box`, `defending_in_box`
- `attacking_near_goal`, `defending_near_goal`
- `total_attacking`, `total_defending`

**Spatial Density**:
- `attacking_density`, `defending_density`
- `numerical_advantage`, `attacker_defender_ratio`

**Positional**:
- `attacking_centroid_x/y`, `defending_centroid_x/y`
- `defending_compactness`, `defending_depth`
- `attacking_to_goal_dist`, `defending_to_goal_dist`
- `keeper_distance_to_goal`

### Pass Technique (3 features)
- `is_inswinging`
- `is_outswinging`
- `is_cross_field_switch`

### Match State (11 features)
- `score_difference`, `match_situation`
- `attacking_team_goals`, `defending_team_goals`
- `minute`, `second`, `period`, `timestamp_seconds`
- `total_subs_before`, `recent_subs_5min`, `minutes_since_last_sub`

### Other (4 features)
- `corner_side`
- `num_attacking_keepers`, `num_defending_keepers`

---

## 5. MODEL PERFORMANCE (CLEAN BASELINE)

### Data Split
- **Train**: 1,546 samples (80%)
- **Test**: 387 samples (20%)
- **Method**: Stratified random split

### Results

| Model | Accuracy | AUC-ROC | MCC | F1 | Precision | Recall |
|-------|----------|---------|-----|-----|-----------|--------|
| **Random Forest** | 68.2% | **0.631** | 0.120 | 0.297 | 41.3% | 23.2% |
| MLP | 70.8% | 0.604 | 0.071 | 0.096 | 46.2% | 5.4% |
| XGBoost | 64.3% | 0.574 | 0.095 | 0.337 | 36.5% | 31.2% |

### Best Model: Random Forest
- **Best AUC-ROC**: 0.631 (best discrimination between classes)
- **Confusion Matrix**:
  ```
           Predicted
           No   Yes
  Actual No  238  37
        Yes  86  26
  ```
- Specificity: 86.5%
- Sensitivity: 23.2%

### Cross-Validation
- RF CV AUC: 0.579 ± 0.032
- XGBoost CV AUC: 0.565 ± 0.032
- MLP CV AUC: 0.565 ± 0.055

---

## 6. FEATURE IMPORTANCE

### Top 10 Random Forest Importance
1. **defending_compactness** (0.065) - Defensive shape tightness
2. **defending_to_goal_dist** (0.063) - Defensive distance to goal
3. **defending_centroid_y** (0.062) - Defense vertical position
4. **keeper_distance_to_goal** (0.060) - GK positioning
5. **defending_depth** (0.058) - Defensive line depth
6. **attacking_centroid_y** (0.057) - Attack vertical position
7. **attacking_centroid_x** (0.056) - Attack horizontal position
8. **timestamp_seconds** (0.054) - When in match
9. **defending_centroid_x** (0.053) - Defense horizontal position
10. **attacking_to_goal_dist** (0.051) - Attacker distance to goal

### Permutation Importance (Impact on AUC)
1. **is_cross_field_switch** (0.047) - Far post corners
2. **is_outswinging** (0.017) - Outswing technique
3. **defending_compactness** (0.013) - Defensive shape

### Statistical Correlations with Shot Outcome
- **is_cross_field_switch**: r=0.132, p<0.001 (strongest)
- **is_outswinging**: r=0.108, p<0.001
- **attacking_centroid_y**: r=0.063, p<0.01

---

## 7. KEY FINDINGS

### 1. Spatial Features Dominate
- Defensive compactness and team centroids are top predictors
- Freeze-frame positioning captures meaningful tactical information

### 2. Pass Technique Matters
- **Cross-field switches** (far post corners) strongly predict shots
- **Outswinging corners** more likely to create opportunities
- These have highest permutation importance

### 3. Better Class Balance Improved Learning
- Previous: 21.8% shots (excluding blocked)
- Current: 29.0% shots (including all outcomes)
- AUC improved from 0.615 to 0.631

### 4. Temporal Context Relevant
- Match timing features (minute, second) in top 10
- May capture fatigue, urgency, tactical changes

### 5. Conservative Prediction
- Models favor specificity over sensitivity
- High false negative rate (miss actual shots)
- Reflects inherent unpredictability of set pieces

---

## 8. METHODS FOR PAPER

### Data Collection
"We used StatsBomb's open event data, extracting 1,933 corner kicks that had associated 360-degree freeze frame data capturing player positions at the moment of corner execution."

### Shot Labeling
"Following TacticAI methodology, we labeled corners as 'shot' if a shot by the attacking team occurred within the next 5 events. All shot outcomes were included: goals, saves, blocked shots, posts, and off-target attempts. This yielded 560 shots (29.0%)."

### Feature Engineering
"We extracted 36 features from freeze-frame data capturing spatial positioning (team centroids, defensive compactness, player counts in box), pass technique (inswinging, outswinging, cross-field switch), and match state (score, timing, substitutions)."

### Leakage Removal
"We identified and removed 7 temporally leaked features that contained information only available after the outcome occurred, including pass landing location (which StatsBomb records as actual, not intended, landing) and shot assist labels."

### Model Training
"We trained Random Forest, XGBoost, and MLP classifiers with class-weighted loss functions to handle the 2.45:1 class imbalance. Models were evaluated using AUC-ROC, accuracy, and MCC on a held-out test set of 387 samples."

---

## 9. RESULTS FOR PAPER

### Main Results
"The best performing model was Random Forest with AUC-ROC of 0.631 and accuracy of 68.2%. This represents the true predictive capability using only information available at the moment of corner execution."

### Feature Importance
"The most predictive features were spatial positioning metrics: defensive compactness (importance=0.065), defensive distance to goal (0.063), and team centroids. Pass technique features, particularly cross-field switches (r=0.132, p<0.001), showed the strongest statistical correlation with shot outcomes."

### Comparison to Literature
"Our 29.0% shot rate aligns with TacticAI's reported ~24% rate, validating our labeling methodology. The moderate AUC-ROC (0.631) reflects the inherent unpredictability of contested set pieces, where factors beyond spatial positioning (player skill, timing, ball physics) determine outcomes."

---

## 10. LIMITATIONS

1. **Limited temporal information** - Freeze frames capture single moment
2. **Missing player attributes** - No height, heading ability, etc.
3. **Conservative predictions** - Models miss most actual shots
4. **Sample size** - 1,933 corners may limit generalization

---

## 11. CODE AND DATA FILES

### Current Data
- `data/processed/corners_features_with_shot.csv` - Main dataset (1,933 × 53)
- `data/processed/corners_with_shot_labels.json` - Shot labels
- `data/processed/corner_labels.csv` - Label file

### Current Models
- `models/clean_baselines/random_forest_clean.pkl`
- `models/clean_baselines/xgboost_clean.pkl`
- `models/clean_baselines/mlp_clean.pkl`
- `models/clean_baselines/clean_features.json` - Feature list
- `models/clean_baselines/clean_baseline_results.json` - Results

### Key Scripts
- `scripts/07_extract_shot_labels.py` - Shot labeling (lines 62-72 for outcomes)
- `scripts/retrain_clean_baselines.py` - Model training
- `scripts/analyze_data_leakage.py` - Leakage analysis
- `scripts/verify_shot_labels.py` - Label verification

---

## 12. SUMMARY TABLE FOR PAPER

| Metric | Value |
|--------|-------|
| Total corners | 1,933 |
| Shot percentage | 29.0% (560) |
| Features | 36 (after removing 7 leaked) |
| Best model | Random Forest |
| AUC-ROC | 0.631 |
| Accuracy | 68.2% |
| MCC | 0.120 |
| Top predictor | is_cross_field_switch (r=0.132) |

---

*Updated: 2025-11-19*
*Use this document for paper writing*
*Ignore all ablation results - they used leaked features*