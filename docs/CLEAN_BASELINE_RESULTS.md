# Clean Baseline Results (Without Leaked Features)

## Summary

All models retrained after removing 7 temporally leaked features. These results represent the **true predictive capability** using only information available at t=0 (corner kick moment).

**Key Change**: Updated `leads_to_shot` to include ALL shot outcomes (including blocked shots, saved off target, etc.)

## Removed Features (7 total)
1. `is_shot_assist` - Directly encodes target
2. `has_recipient` - Only known after pass completes
3. `duration` - Time until next event
4. `pass_end_x/y` - Actual landing location (not intended)
5. `pass_length/angle` - Computed from actual landing

## Clean Feature Set
**36 features** remaining:
- Freeze-frame positions (18 features)
- Pass technique (3 features)
- Score state (4 features)
- Temporal (4 features)
- Substitutions (3 features)
- Other (4 features)

## Dataset Statistics

- **Total samples**: 1,933 corners
- **Shots**: 560 (29.0%)
- **No shots**: 1,373 (71.0%)
- **Imbalance ratio**: 2.45:1

## Model Performance Comparison

| Model | Accuracy | AUC-ROC | MCC | F1 Score | Precision | Recall |
|-------|----------|---------|-----|----------|-----------|--------|
| **Random Forest** | 68.2% | 0.631 | 0.120 | 0.297 | 0.413 | 0.232 |
| **MLP** | 70.8% | 0.604 | 0.071 | 0.096 | 0.462 | 0.054 |
| **XGBoost** | 64.3% | 0.574 | 0.095 | 0.337 | 0.365 | 0.312 |

### Best Model: Random Forest
- **AUC-ROC**: 0.631 (best discrimination)
- **Accuracy**: 68.2%
- **MCC**: 0.120

## Confusion Matrices

### Random Forest (Best AUC)
```
         Predicted
         No   Yes
Actual No  238  37
      Yes  86  26
```
- Specificity: 86.5%
- Sensitivity: 23.2%
- Balanced predictor

### XGBoost (Best Recall)
```
         Predicted
         No   Yes
Actual No  214  61
      Yes  77  35
```
- Better recall (31.2%)
- More false positives
- Better for shot detection

### MLP (Failed to Learn)
```
         Predicted
         No   Yes
Actual No  268   7
      Yes  106   6
```
- Very conservative (5.4% recall)
- Needs hyperparameter tuning

## Comparison with Previous Results

### Before Including Blocked Shots
- Shots: 422 (21.8%)
- RF: 78.8% accuracy, 0.615 AUC

### After Including Blocked Shots
- Shots: 560 (29.0%)
- RF: 68.2% accuracy, 0.631 AUC

**Note**: Lower accuracy but better AUC due to more balanced classes.

## Key Insights

1. **Better Class Balance**
   - Previous: 21.8% shots (imbalance 3.58:1)
   - Current: 29.0% shots (imbalance 2.45:1)
   - More balanced → better learning

2. **Improved AUC**
   - Random Forest AUC improved from 0.615 to 0.631
   - Better discrimination between classes

3. **Label Agreement with StatsBomb**
   - Only 1 disagreement with `is_shot_assist` now
   - 190 additional indirect assists captured

4. **Top Predictive Features**
   - `is_cross_field_switch` (far post corners)
   - `is_outswinging` (outswing technique)
   - `defending_compactness` (defensive shape)

## Files Generated
- `/models/clean_baselines/mlp_clean.pkl`
- `/models/clean_baselines/random_forest_clean.pkl`
- `/models/clean_baselines/xgboost_clean.pkl`
- `/models/clean_baselines/clean_features.json`
- `/models/clean_baselines/clean_baseline_results.json`

## Recommendations

### For Your Paper
1. **Report**: 68.2% accuracy, 0.631 AUC with clean features
2. **Emphasize**: Better AUC with balanced classes
3. **Note**: Includes all shot types (blocked, saved, goals, etc.)

### For Model Improvement
1. **Adjust class weights** for XGBoost
2. **Hyperparameter tuning** for MLP
3. **Ensemble** Random Forest + XGBoost
4. **Feature engineering**: defensive vulnerability metrics

---
*Generated: 2025-11-19*
*Clean features: 36*
*Training samples: 1,546*
*Test samples: 387*
*Shot percentage: 29.0%*