# CornerTactics: Paper Methods and Results Summary

**Updated November 21, 2025**

---

## CURRENT STATUS

Most previous results contained temporal data leakage and have been removed. The only valid results are:
- **True performance**: 71% accuracy with 19 temporally valid features
- **AUC**: 0.52 (barely better than random)
- See `docs/DATA_LEAKAGE_FINDINGS.md` for details

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

## 4. VALID FEATURE SET (22 Features)

These are the ONLY temporally valid features available at the moment of corner kick execution:

### Event Data (7 features)
- `second` - Match second when corner awarded
- `minute` - Match minute
- `period` - First or second half
- `corner_x` - Corner position X (always 120 or 0)
- `corner_y` - Corner position Y (always 80 or 0)
- `attacking_team_goals` - Goals scored by attacking team
- `defending_team_goals` - Goals conceded (defending team's score)

### 360° Freeze Frame Data (15 features)
All freeze frame features capture player positions AT the moment of corner kick:

**Player Counts**:
- `total_attacking` - Total attacking players
- `total_defending` - Total defending players
- `attacking_in_box` - Attackers in penalty area
- `defending_in_box` - Defenders in penalty area
- `attacking_near_goal` - Attackers near goal
- `defending_near_goal` - Defenders near goal

**Spatial Metrics**:
- `attacking_density` - Spatial concentration of attackers
- `defending_density` - Spatial concentration of defenders
- `numerical_advantage` - Attackers minus defenders
- `attacker_defender_ratio` - Ratio of attackers to defenders

**Positional Distances**:
- `defending_depth` - Defensive line Y position
- `attacking_to_goal_dist` - Average attacker distance to goal
- `defending_to_goal_dist` - Average defender distance to goal
- `keeper_distance_to_goal` - Goalkeeper distance to goal

**Other**:
- `corner_side` - Left (0) or right (1) corner

---

## 5. MODEL PERFORMANCE (NO TEMPORAL LEAKAGE)

### Data Split
- **Total samples**: 1,933 corners
- **Train**: 1,546 samples (80%)
- **Test**: 387 samples (20%)
- **Method**: Stratified random split
- **Class distribution**: 560 shots (29.0%), 1,373 no shots (71.0%)

### Valid Results (22 Temporally Valid Features)

| Model | Test Accuracy | Test AUC | CV Accuracy | CV AUC |
|-------|---------------|----------|-------------|---------|
| **MLP (Best)** | **71.06%** | **0.556** | 61.35% ± 2.77% | 0.530 ± 0.031 |
| Random Forest | 62.79% | 0.510 | - | - |
| XGBoost | 61.50% | 0.545 | - | - |

### Key Insights
- **Performance barely better than random** (AUC ~0.55)
- **Baseline accuracy** (predicting all "no shot"): ~71%
- **MLP improvement over baseline**: Only ~0.06%
- **MLP performs best** but still limited predictive power
- **Cross-validation lower than test** (61.35% vs 71.06%) suggests favorable test set
- **Corner outcomes are inherently unpredictable** from pre-kick features alone
- **True predictability is very limited** - execution matters more than setup

---

## 6. FEATURE IMPORTANCE

**NOTE:** With 22 valid features and AUC ~0.55, feature importance is limited. The weak predictive power means no single feature strongly discriminates outcomes.

### Model Characteristics
- **MLP**: Best overall (71.06% accuracy, 0.556 AUC)
- **Random Forest**: Second best (62.79% accuracy, 0.510 AUC)
- **XGBoost**: Third (61.50% accuracy, 0.545 AUC)

### Top 10 Most Important Features (Random Forest)
1. **attacking_to_goal_dist** (0.131) - Avg attacker distance to goal
2. **keeper_distance_to_goal** (0.114) - GK positioning
3. **defending_to_goal_dist** (0.111) - Avg defender distance to goal
4. **defending_depth** (0.110) - Defensive line position
5. **minute** (0.090) - Match minute
6. **second** (0.085) - Match second
7. **attacker_defender_ratio** (0.042) - Attacker/defender ratio
8. **defending_near_goal** (0.041) - Defenders near goal
9. **total_attacking** (0.035) - Total attackers
10. **attacking_near_goal** (0.035) - Attackers near goal

### Key Observations
1. **All models perform near baseline** - Predicting "no shot" always gives ~71% accuracy
2. **Limited discrimination** - AUC ~0.52 is barely better than random (0.50)
3. **Freeze frame features most relevant** - 12 of 19 features capture player positioning
4. **Temporal features less important** - Match timing provides minimal signal
5. **Defensive positioning** - Features like `defending_depth` and `defending_to_goal_dist` show slight relevance

---

## 7. KEY FINDINGS

### 1. Corner Kick Outcomes Are Largely Unpredictable
- **AUC ~0.52** indicates near-random prediction
- Pre-kick positioning provides minimal predictive signal
- **Execution matters far more than setup**

### 2. Limited Value of Freeze Frame Data
- Despite 12 positioning features, models achieve only marginal improvement
- Suggests that small variations in delivery dominate outcomes
- Player skill and in-the-moment decisions more important than positioning

### 3. Baseline Comparison
- Predicting "no shot" always: ~71% accuracy
- Best model (MLP): 71.06% accuracy
- **Improvement: Only 0.06%** - practically negligible

### 4. Cross-Validation Reveals Overfitting
- Test accuracy: 71.06%
- CV accuracy: 61.35%
- Suggests test set was somewhat favorable
- True performance likely closer to 61%

### 5. Previous High Performance Was Due to Leakage
- Original 87.97% accuracy used leaked features
- Removed 9 temporally invalid features
- Performance dropped to realistic 71%
- **Leakage was responsible for 16.91% of apparent accuracy**

---

## 8. METHODS FOR PAPER

### Data Collection
"We used StatsBomb's open event data, extracting 1,933 corner kicks that had associated 360-degree freeze frame data capturing player positions at the moment of corner execution."

### Shot Labeling
"Following TacticAI methodology, we labeled corners as 'shot' if a shot by the attacking team occurred within the next 5 events. All shot outcomes were included: goals, saves, blocked shots, posts, and off-target attempts. This yielded 560 shots (29.0%)."

### Feature Engineering
"We extracted 22 temporally valid features from event data and 360° freeze-frame data. Features included match timing (minute, second, period), corner position, score state (both teams' goals), and 15 freeze-frame metrics capturing player positioning at the moment of corner execution (player counts, spatial density, numerical advantage, distances to goal, goalkeeper position)."

### Leakage Removal
"We identified and removed 9 temporally leaked features that contained information only available after the outcome occurred, including pass landing location (which StatsBomb records as actual, not intended, landing position), shot assist labels, pass duration, cross-field switch indicator, and recipient information. Only features available at t=0 (corner kick moment) were retained."

### Model Training
"We trained Random Forest, XGBoost, and MLP classifiers with class-weighted loss functions to handle the 2.45:1 class imbalance. Models were evaluated using AUC-ROC, accuracy, and MCC on a held-out test set of 387 samples."

---

## 9. RESULTS FOR PAPER

### Main Results
"The best performing model was MLP with accuracy of 71.06% and AUC-ROC of 0.556. However, this represents only a 0.06% improvement over the baseline (predicting 'no shot' for all instances gives 71% accuracy). The near-random AUC (~0.55) indicates that corner kick outcomes are largely unpredictable using only information available at the moment of corner execution."

### Limited Predictive Power
"All three models (MLP, Random Forest, XGBoost) achieved AUC values between 0.510 and 0.556, barely exceeding random chance (0.50). Cross-validation accuracy (61.35%) was significantly lower than test accuracy (71.06%), suggesting limited generalization. These results indicate that pre-kick positioning provides minimal predictive signal for corner kick outcomes."

### Comparison to Literature
"Our 29.0% shot rate aligns with TacticAI's reported ~24% rate, validating our labeling methodology. The low AUC-ROC (0.55) reflects the inherent unpredictability of contested set pieces, where factors beyond spatial positioning (player skill, execution quality, ball physics, split-second decisions) dominate outcomes. This finding suggests that predictive models for corner kicks should focus on execution prediction rather than outcome prediction."

---

## 10. LIMITATIONS

1. **Limited temporal information** - Freeze frames capture single moment, not player movements
2. **Missing player attributes** - No height, heading ability, historical performance metrics
3. **Inherent unpredictability** - Corner outcomes dominated by execution quality, not setup
4. **Sample size** - 1,933 corners with 360° data limits generalization
5. **Favorable test set** - CV accuracy (59.85%) lower than test (71.32%) suggests overfitting
6. **Near-baseline performance** - 0.3% improvement over majority class baseline is negligible

---

## 11. CODE AND DATA FILES

### Valid Data (Use These)
- `data/processed/corners_features_temporal_valid.csv` - Clean dataset (1,933 × 26: 22 features + metadata)
- `data/processed/temporal_valid_features.json` - Feature metadata (22 features)
- `data/processed/corner_labels.csv` - Shot labels

### Valid Results
- `results/no_leakage/training_results_no_leakage.json` - Model performance with 19 features

### Key Scripts
- `scripts/14_extract_temporally_valid_features.py` - Extract clean 19 features
- `scripts/15_retrain_without_leakage.py` - Train models without leakage
- `scripts/analyze_data_leakage.py` - Leakage detection analysis

---

## 12. SUMMARY TABLE FOR PAPER

| Metric | Value |
|--------|-------|
| Total corners | 1,933 |
| Shot percentage | 29.0% (560) |
| No-shot percentage | 71.0% (1,373) |
| Valid features | 22 |
| Leaked features removed | 9 |
| Best model | MLP |
| Test accuracy | 71.06% |
| Test AUC-ROC | 0.556 |
| CV accuracy | 61.35% ± 2.77% |
| CV AUC-ROC | 0.530 ± 0.031 |
| Baseline (predict "no shot") | ~71% |
| Improvement over baseline | +0.06% |
| Interpretation | **Near-random prediction** |

---

*Updated: 2025-11-25*
*Use this document for paper writing*
*All results validated with 22 temporally valid features*
*Previous high-accuracy results (87.97%) were due to data leakage*