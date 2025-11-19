# Clean Baseline Results (Without Leaked Features)

## Summary

All models retrained after removing 8 temporally leaked features. These results represent the **true predictive capability** using only information available at t=0 (corner kick moment).

## Removed Features
1. `is_shot_assist` - Directly encodes target
2. `has_recipient` - Only known after pass completes
3. `duration` - Time until next event
4. `pass_end_x/y` - Actual landing location (not intended)
5. `pass_length/angle` - Computed from actual landing
6. `pass_outcome_id/encoded` - Pass completion status

## Clean Feature Set
**36 features** remaining:
- Freeze-frame positions (18 features)
- Pass technique (3 features)
- Score state (4 features)
- Temporal (4 features)
- Substitutions (3 features)
- Other (4 features)

## Model Performance Comparison

| Model | Accuracy | AUC-ROC | MCC | F1 Score | Precision | Recall |
|-------|----------|---------|-----|----------|-----------|--------|
| **Random Forest** | 78.8% | 0.615 | 0.167 | 0.163 | 0.571 | 0.095 |
| **MLP** | 78.0% | 0.599 | -0.027 | 0.000 | 0.000 | 0.000 |
| **XGBoost** | 74.7% | 0.593 | 0.187 | 0.338 | 0.391 | 0.298 |

### Best Model: Random Forest
- **Accuracy**: 78.8%
- **AUC-ROC**: 0.615
- **MCC**: 0.167

## Confusion Matrices

### Random Forest (Best Overall)
```
         Predicted
         No   Yes
Actual No  297   6
      Yes  76   8
```
- High specificity (98.0%)
- Low sensitivity (9.5%)
- Conservative predictor

### XGBoost (Best Balance)
```
         Predicted
         No   Yes
Actual No  264  39
      Yes  59  25
```
- Better recall (29.8%)
- More false positives
- Better for shot detection

### MLP (Failed to Learn)
```
         Predicted
         No   Yes
Actual No  302   1
      Yes  84   0
```
- Predicts almost all as "No Shot"
- Needs hyperparameter tuning

## Key Insights

1. **Performance Drop from Leakage Removal**
   - Expected: ~82% → 78.8% (actual)
   - Confirms ~3-4% was from leaked features

2. **Class Imbalance Challenge**
   - 21.8% shots in dataset
   - Models struggle with minority class
   - Random Forest too conservative (9.5% recall)

3. **Feature Importance (from earlier analysis)**
   - Top features: defensive compactness, centroids, outswinging corners
   - Spatial freeze-frame features dominate

## Recommendations

### For Your Paper
1. **Report these clean results** as your baseline
2. **Acknowledge class imbalance** as a limitation
3. **Emphasize spatial features** as key predictors

### For Model Improvement
1. **Adjust class weights** more aggressively
2. **Try SMOTE** or other oversampling
3. **Ensemble** Random Forest + XGBoost
4. **Feature engineering**: defensive vulnerability metrics

### Next Steps
1. Hyperparameter optimization on clean features
2. Ensemble methods
3. Deep learning with attention on spatial features

## Files Generated
- `/models/clean_baselines/mlp_clean.pkl`
- `/models/clean_baselines/random_forest_clean.pkl`
- `/models/clean_baselines/xgboost_clean.pkl`
- `/models/clean_baselines/clean_features.json`
- `/models/clean_baselines/clean_baseline_results.json`

---
*Generated: 2025-11-19*
*Clean features: 36*
*Training samples: 1,546*
*Test samples: 387*