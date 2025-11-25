# CornerTactics - Verified Valid State Summary

**Date**: 2025-11-25
**Status**: VERIFIED - All documentation now consistent

---

## CURRENT VALID STATE

### Dataset
- **Total corners**: 1,933 with 360 freeze frame data
- **Shot percentage**: 29.0% (560 shots)
- **No-shot percentage**: 71.0% (1,373 no shots)
- **Data file**: `data/processed/corners_features_temporal_valid.csv`

### Valid Features: 22 Total

#### Event Data (7 features)
1. `second` - Match second
2. `minute` - Match minute
3. `period` - Half (1 or 2)
4. `corner_x` - Corner X position
5. `corner_y` - Corner Y position
6. `attacking_team_goals` - Goals scored by attacking team
7. `defending_team_goals` - Goals conceded (defending team's score)

#### Freeze Frame Data (15 features)
8. `total_attacking` - Total attacking players
9. `total_defending` - Total defending players
10. `attacking_in_box` - Attackers in penalty box
11. `defending_in_box` - Defenders in penalty box
12. `attacking_near_goal` - Attackers near goal
13. `defending_near_goal` - Defenders near goal
14. `attacking_density` - Attacking spatial concentration
15. `defending_density` - Defending spatial concentration
16. `numerical_advantage` - Attackers minus defenders
17. `attacker_defender_ratio` - Ratio of attackers to defenders
18. `corner_side` - Left (0) or right (1)
19. `defending_depth` - Defensive line position
20. `attacking_to_goal_dist` - Avg attacker distance to goal
21. `defending_to_goal_dist` - Avg defender distance to goal
22. `keeper_distance_to_goal` - Goalkeeper distance to goal

### Valid Model Results

| Model | Test Acc | Test AUC | CV Acc | CV AUC |
|-------|----------|----------|--------|--------|
| **MLP (Best)** | **71.06%** | **0.556** | 61.35% | 0.530 |
| XGBoost | 61.50% | 0.545 | - | - |
| Random Forest | 62.79% | 0.510 | - | - |

### Key Interpretation
- **Baseline** (always predict "no shot"): ~71% accuracy
- **Best model improvement**: +0.06% over baseline
- **AUC ~0.55**: Limited predictive power (random = 0.50)
- **Conclusion**: Corner outcomes are largely unpredictable from pre-kick features

---

## REMOVED LEAKED FEATURES (9 total)

These features were removed because they contain information only available AFTER the corner:

1. `is_shot_assist` - Literally the prediction target
2. `has_recipient` - Only known after pass completes
3. `duration` - Time until next event
4. `pass_end_x` - Actual landing X (not intended)
5. `pass_end_y` - Actual landing Y (not intended)
6. `pass_length` - Calculated from actual landing
7. `pass_angle` - Calculated from actual landing
8. `pass_outcome` - Pass outcome status
9. `is_cross_field_switch` - Whether ball switched sides (requires knowing landing)

---

## VALID FILES TO USE

### Data
- `data/processed/corners_features_temporal_valid.csv` - 22 valid features
- `data/processed/temporal_valid_features.json` - Feature metadata

### Results
- `results/no_leakage/training_results_no_leakage.json` - Valid model performance

### Documentation
- `docs/CURRENT_VALID_RESULTS.md` - **START HERE**
- `docs/VERIFICATION_SUMMARY.md` - **This file - Quick reference**
- `docs/PAPER_METHODS_AND_RESULTS.md` - Updated 2025-11-25 with correct results
- `docs/DATA_LEAKAGE_FINDINGS.md` - Why results changed
- `docs/FEATURE_REMOVAL_METHODOLOGY.md` - How features were validated

### Scripts
- `scripts/14_extract_temporally_valid_features.py` - Extract valid features
- `scripts/15_retrain_without_leakage.py` - Train valid models

---

## INVALID FILES (DO NOT USE)

### Marked as Invalid
- `docs/CLEAN_BASELINE_RESULTS.md` - Uses 36 features (includes leaks), reports 68.2% accuracy (WRONG)
- `notes/features/data-leakage-analysis.md` - Uses 36 features, invalid results

These files have **strong warnings at the top** but are kept for historical context.

---

## VERIFICATION CHECKLIST

- Verified 22 features in temporal_valid.csv
- Confirmed results match training_results_no_leakage.json
- Updated PAPER_METHODS_AND_RESULTS.md with correct metrics
- Updated CURRENT_VALID_RESULTS.md with correct metrics
- Added strong warnings to invalid files
- Documented all leaked features (9 total)
- Cross-referenced all documentation files
- Verified no conflicting metrics remain

---

## FINAL TRUTH

**For any paper, publication, or decision-making, use ONLY:**
- **22 valid features**
- **MLP: 71.06% accuracy, 0.556 AUC**
- **Conclusion: Near-random predictive power**

Any results showing >75% accuracy are likely contaminated with data leakage.

---

*Verification completed: 2025-11-25*
*All documentation now consistent and accurate*
