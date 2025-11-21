# Current Valid Results - CornerTactics

**Date:** November 21, 2025
**Status:** These are the only valid results after removing all temporal data leakage

---

## The Truth About Corner Kick Prediction

After extensive analysis and removal of temporal data leakage, we've discovered that **corner kick outcomes are largely unpredictable** using only information available at the time of the kick.

### Valid Performance Metrics

| Model | Test Accuracy | Test AUC | CV Accuracy | CV AUC |
|-------|---------------|----------|-------------|---------|
| **MLP (Best)** | 71.32% | 0.521 | 59.85% ± 1.82% | 0.529 ± 0.018 |
| Random Forest | 63.57% | 0.505 | - | - |
| XGBoost | 60.47% | 0.509 | - | - |

**Key Insight:** AUC of ~0.52 means the models are barely better than random guessing.

---

## Valid Features (19 Total)

These are the ONLY features that can legitimately be used for prediction:

### Event Data (7 features)
- `second` - When in the match
- `minute` - Match minute
- `period` - First or second half
- `corner_x` - Always ~120 or ~0
- `corner_y` - Always ~80 or ~0
- `team_id` - Team taking corner (if available)
- `player_id` - Corner taker (if available)

### Freeze Frame Data (12 features)
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

- **Previous claim (with leaks):** 87.97% accuracy
- **Reality (no leaks):** 71.32% accuracy
- **Baseline (always predict no shot):** 71% accuracy
- **Improvement over baseline:** ~0.3%

### 3. Corner Kicks Are Chaotic Events

The fact that we can barely beat random guessing suggests that:
- Execution matters more than setup
- Small variations in delivery create large outcome differences
- Player skill/decisions in the moment dominate pre-kick positioning

---

## Files and Scripts

### Valid Data Files
- `/data/processed/corners_features_temporal_valid.csv` - Clean dataset
- `/data/processed/temporal_valid_features.json` - Feature list

### Valid Scripts
- `scripts/14_extract_temporally_valid_features.py` - Extract valid features
- `scripts/15_retrain_without_leakage.py` - Train models properly

### Valid Documentation
- This file - Current valid results
- `docs/DATA_LEAKAGE_FINDINGS.md` - Detailed leakage analysis
- `docs/TEMPORAL_DATA_LEAKAGE_ANALYSIS.md` - Feature-by-feature analysis
- `docs/FEATURE_REMOVAL_METHODOLOGY.md` - **How we determined which features to remove**

---

## Deleted Files (Contained Invalid Results)

The following have been permanently deleted as they contained leaked results:

### Documentation
- `docs/OPTIMAL_FEATURE_SELECTION.md` (claimed 87.97% accuracy)
- `docs/ABLATION_STUDY_*.md` (all based on leaked features)

### Scripts
- All ablation study scripts (09-13)
- Feature extraction scripts with leaks
- Optimal feature search scripts

### Results
- `/results/optimal_search/` directory
- `/results/ablation/` directory

---

## Recommendations

### For Research
1. **Accept the reality:** Corner outcomes are largely random from pre-kick features
2. **Focus on different problems:**
   - Predicting where the ball will be delivered (execution)
   - Predicting defensive clearance zones
   - Optimizing player positioning

### For Practice
1. **Execution > Setup:** Focus on delivery quality over positioning
2. **Rehearsed plays:** Since prediction is hard, use set routines
3. **React quickly:** Success comes from adapting during the play

---

## Conclusion

The discovery of temporal data leakage has been humbling but valuable. We've learned that:

1. **Corner kick outcomes are inherently unpredictable** (AUC ~0.52)
2. **Many published results may contain similar leakage**
3. **The problem is harder than initially thought**

This is good science - we've moved from incorrect certainty to accurate understanding.

---

*"The best models are not those that achieve the highest accuracy, but those that achieve their accuracy honestly."*