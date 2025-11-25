# Current Valid Results - CornerTactics

**Date:** November 25, 2025
**Status:** These are the only valid results after removing all temporal data leakage

---

## The Truth About Corner Kick Prediction

After extensive analysis and removal of temporal data leakage, we've discovered that **corner kick outcomes are largely unpredictable** using only information available at the time of the kick.

---

## Experimental Setup

### Data Split (Match-Based)
To prevent data leakage from same-match corners appearing in different sets:

| Set | Samples | Percentage |
|-----|---------|------------|
| Train | 1,155 | 59.8% |
| Validation | 371 | 19.2% |
| Test | 407 | 21.1% |
| **Total** | **1,933** | 100% |

Splits are stratified by outcome and ensure no match overlap between sets.

### Prediction Tasks
1. **Binary Shot Prediction**: Did the corner lead to a shot? (Yes/No)
2. **Multi-Class Outcome Prediction**: What was the immediate outcome? (4 classes)

---

## Results: Binary Shot Prediction

| Model | Train Acc | Val Acc | Test Acc | Val AUC | Test AUC |
|-------|-----------|---------|----------|---------|----------|
| Random Forest | 92.29% | 66.85% | 59.95% | 0.5497 | 0.4526 |
| XGBoost | 99.39% | 63.61% | 60.44% | 0.5150 | 0.5095 |
| MLP | 70.48% | 72.78% | 70.52% | 0.5889 | 0.4324 |
| **Baseline** | - | - | **71.74%** | - | 0.5000 |

**Key Findings:**
- All models perform at or below the baseline (always predict "no shot")
- Test AUC values around 0.45-0.51 indicate no predictive power (random = 0.50)
- Severe overfitting: Random Forest and XGBoost achieve 92-99% train accuracy but fail on test
- Shot rate in training data: 30.0%

---

## Results: Multi-Class Outcome Prediction

### Classes
| Class | Description | Test Count |
|-------|-------------|------------|
| Ball Receipt | Attacking team receives the ball | 216 (53.1%) |
| Clearance | Defending team clears | 97 (23.8%) |
| Goalkeeper | Keeper collects/saves | 44 (10.8%) |
| Other | Duel, foul, etc. | 50 (12.3%) |

### Performance

| Model | Train Acc | Val Acc | Test Acc | Val F1 | Test F1 | Test AUC |
|-------|-----------|---------|----------|--------|---------|----------|
| Random Forest | 88.48% | 37.74% | 43.00% | 0.2624 | 0.2819 | 0.5219 |
| XGBoost | 98.53% | 49.06% | 49.39% | 0.2327 | 0.2237 | 0.5291 |
| MLP | 57.49% | 53.37% | 50.86% | 0.1979 | 0.1734 | 0.4995 |
| **Baseline** | - | - | **53.07%** | - | - | - |

### Classification Report (Best Model: MLP)
```
              precision    recall  f1-score   support

Ball Receipt       0.52      0.95      0.68       216
   Clearance       0.08      0.01      0.02        97
  Goalkeeper       0.00      0.00      0.00        44
       Other       0.00      0.00      0.00        50

    accuracy                           0.51       407
   macro avg       0.15      0.24      0.17       407
weighted avg       0.30      0.51      0.36       407
```

**Key Findings:**
- All models perform at or below the majority-class baseline (53.07%)
- Models essentially learn to predict "Ball Receipt" for everything
- Macro F1 scores of 0.17-0.28 indicate poor performance across all classes
- Minority classes (Goalkeeper, Other) are completely ignored

---

## Valid Features (22 Total)

These are the ONLY features that can legitimately be used for prediction:

### Event Data (7 features)
- `second` - When in the match
- `minute` - Match minute
- `period` - First or second half
- `corner_x` - Corner location x
- `corner_y` - Corner location y
- `attacking_team_goals` - Goals scored by attacking team
- `defending_team_goals` - Goals conceded (defending team's score)

### Freeze Frame Data (15 features)
- `total_attacking` - Attacking players count
- `total_defending` - Defending players count
- `attacking_in_box` - Attackers in penalty area
- `defending_in_box` - Defenders in penalty area
- `attacking_near_goal` - Attackers near goal
- `defending_near_goal` - Defenders near goal
- `attacking_density` - Spatial concentration
- `defending_density` - Defensive concentration
- `numerical_advantage` - Attacker-defender difference
- `attacker_defender_ratio` - Ratio of attackers to defenders
- `defending_depth` - Defensive line position
- `attacking_to_goal_dist` - Avg attacker distance to goal
- `defending_to_goal_dist` - Avg defender distance to goal
- `keeper_distance_to_goal` - Goalkeeper distance to goal
- `corner_side` - Left (0) or right (1) corner

---

## What We Learned

### 1. Most "Predictive" Features Were Leaks

The features that seemed most predictive were actually outcomes:
- `is_shot_assist` - Literally tells you if corner led to shot
- `pass_end_x/y` - Where ball actually landed (not intended target)
- `duration` - How long the play lasted (only known after)
- `pass_length` - Actual distance traveled

### 2. True Predictability is Very Limited

| Metric | With Leaks | Without Leaks | Baseline |
|--------|------------|---------------|----------|
| Binary Shot Accuracy | 87.97% | 70.52% | 71.74% |
| Multi-Class Accuracy | - | 50.86% | 53.07% |

Models perform **at or below baseline** when leakage is removed.

### 3. Corner Kicks Are Chaotic Events

The fact that we can barely beat random guessing suggests that:
- Execution matters more than setup
- Small variations in delivery create large outcome differences
- Player skill/decisions in the moment dominate pre-kick positioning

---

## Files and Scripts

### Valid Data Files
- `data/processed/corners_features_temporal_valid.csv` - Clean dataset (22 features)
- `data/processed/temporal_valid_features.json` - Feature list
- `data/processed/train_indices.csv` - Training set indices
- `data/processed/val_indices.csv` - Validation set indices
- `data/processed/test_indices.csv` - Test set indices

### Valid Scripts
- `scripts/14_extract_temporally_valid_features.py` - Extract valid features
- `scripts/15_retrain_without_leakage.py` - Train models with proper splits

### Results
- `results/no_leakage/training_results.json` - Full results JSON

---

## Conclusions

1. **Corner kick outcomes are inherently unpredictable** from pre-kick information
2. **Binary shot prediction**: AUC ~0.45-0.51 (no better than random)
3. **Multi-class prediction**: Models just predict majority class
4. **Overfitting is severe**: High train accuracy, poor generalization
5. **The problem is harder than initially thought**

---

## Recommendations

### For Research
1. Accept that pre-kick features alone cannot predict outcomes
2. Consider different problems:
   - Predicting delivery location (where will the ball go?)
   - Optimal positioning recommendations
   - Post-delivery outcome prediction (with trajectory info)

### For Practice
1. **Execution > Setup:** Focus on delivery quality over positioning
2. **Rehearsed plays:** Since prediction is hard, use set routines
3. **React quickly:** Success comes from adapting during the play

---

*"The best models are not those that achieve the highest accuracy, but those that achieve their accuracy honestly."*
