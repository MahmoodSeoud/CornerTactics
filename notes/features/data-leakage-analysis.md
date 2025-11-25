# ⚠️ [DEPRECATED - DO NOT USE] Data Leakage Analysis

> **⚠️ THIS FILE CONTAINS INVALID RESULTS ⚠️**
>
> **Problem**: This analysis used 36 features including leaked features like `is_cross_field_switch`
>
> **Correct Analysis**: See `docs/FEATURE_REMOVAL_METHODOLOGY.md` for valid 19-feature analysis
>
> **Correct Results**: See `docs/CURRENT_VALID_RESULTS.md` for accurate performance (71% accuracy, 0.52 AUC)
>
> **DO NOT USE THIS FILE FOR ANY DECISION MAKING**

## Context
- **Task**: Binary classification predicting shot outcomes from corner kicks
- **Dataset**: 1,933 corner kicks from StatsBomb
- **Current Distribution**: 560 shots (29.0%), 1,373 no shots (71.0%)
- **Problem**: Suspected temporal data leakage in features

## Label Definition Update

### leads_to_shot Now Includes ALL Shot Outcomes
- Goal
- Saved
- Post
- Off Target (Off T)
- Wayward
- Blocked
- Saved Off Target
- Saved to Post

**Result**: 560 shots (29.0%) vs previous 422 (21.8%)

## Critical Discovery: pass_end_x/y ARE LEAKED

### Evidence from StatsBomb Event Analysis
Analyzed actual event JSON files and confirmed:
```python
# Corner with recipient:
Corner location: [120.0, 80.0]
Pass end_location: [114.5, 75.0]
Next event: Ball Receipt at [114.5, 75.0]  # EXACT MATCH!
```

**Conclusion**: `pass_end_x/y` represents WHERE THE BALL ACTUALLY LANDED, not intended target.

## Complete Leakage Analysis Results

### Leaked Features Removed (7 total)
1. **is_shot_assist** (MCC=0.753) - Directly encodes if next event is shot
2. **has_recipient** (MCC=0.303) - Only known after pass completes
3. **pass_end_x** (MCC=-0.143) - Actual landing location
4. **pass_end_y** - Actual landing location
5. **pass_length** (MCC=0.146) - Computed from actual landing
6. **pass_angle** (MCC=-0.083) - Computed from actual landing
7. **duration** (MCC=0.157) - Time until next event

## Label Verification: leads_to_shot vs is_shot_assist

### Cross-tabulation
```
                    leads_to_shot=0   leads_to_shot=1
is_shot_assist=0         1372              190
is_shot_assist=1            1              370
```

### Discrepancies Explained
- **1 case** where is_shot_assist=1 but leads_to_shot=0: Shot after 5-event window
- **190 cases** where is_shot_assist=0 but leads_to_shot=1: Indirect assists (headers, scrambles)

## ⚠️ DEPRECATED: This analysis used 36 features including leaked features

**See `docs/FEATURE_REMOVAL_METHODOLOGY.md` for the correct 19 temporally valid features.**

This file contains outdated analysis that included leaked features like:
- `is_cross_field_switch` (99.6% correlated with outcome)
- Pass technique features that may be outcome-based
- Other temporally invalid features

**Do not use this analysis for model evaluation.**

## Feature Importance Analysis

### Top 10 Most Important Features (Random Forest)
1. **defending_compactness** (0.065) - How tight defensive shape is
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
- **is_cross_field_switch**: r=0.132, p<0.001 (strongest predictor)
- **is_outswinging**: r=0.108, p<0.001
- **attacking_centroid_y**: r=0.063, p<0.01

## Clean Baseline Model Performance

| Model | Accuracy | AUC-ROC | MCC | F1 | Precision | Recall |
|-------|----------|---------|-----|-----|-----------|--------|
| **Random Forest** | 68.2% | 0.631 | 0.120 | 0.297 | 41.3% | 23.2% |
| **MLP** | 70.8% | 0.604 | 0.071 | 0.096 | 46.2% | 5.4% |
| **XGBoost** | 64.3% | 0.574 | 0.095 | 0.337 | 36.5% | 31.2% |

### Best Model: Random Forest
- Best AUC-ROC (0.631)
- Balanced precision/recall
- Confusion Matrix: TN=238, FP=37, FN=86, TP=26

### Most Balanced: XGBoost
- Best recall (31.2%)
- Confusion Matrix: TN=214, FP=61, FN=77, TP=35
- Better for finding shots

## Key Insights

1. **Spatial features dominate** - Defensive compactness and team centroids are top predictors
2. **Technique matters** - Cross-field switches and outswinging corners predict shots
3. **Better class balance** - 29% shots vs previous 21.8% improves learning
4. **AUC improved** - 0.631 vs previous 0.615 with better balanced data

## Files and Scripts Created

### Analysis Scripts
- `scripts/analyze_data_leakage.py` - Comprehensive leakage detection
- `scripts/feature_importance_simple.py` - Feature analysis
- `scripts/retrain_clean_baselines.py` - Retrain all models
- `scripts/train_xgboost_only.py` - XGBoost training fix
- `scripts/verify_shot_labels.py` - Label verification

### Results
- `/models/clean_baselines/` - All retrained models
- `/results/feature_importance/` - Feature analysis results
- `/reports/data_leakage_report.md` - Leakage findings

### Documentation
- `/docs/CLEAN_BASELINE_RESULTS.md` - Clean model performance
- `/docs/SHOT_LABEL_VERIFICATION.md` - Label verification

## Environment Fix
NumPy had missing OpenBLAS dependency. Fixed by:
```bash
module load GCC/13.2.0 OpenBLAS/0.3.24-GCC-13.2.0
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH
```

## Recommendations for Paper

### Must State
1. "Removed temporal leakage features only available post-outcome"
2. "pass_end location represents actual landing, not intended target"
3. "Clean baseline: 68.2% accuracy, 0.631 AUC using only t=0 information"
4. "Includes all shot types: goals, saves, blocks, posts, off-target"

### Emphasize
1. Spatial freeze-frame features are primary predictors
2. Defensive compactness and cross-field switches key factors
3. 29% shot rate with all outcomes included

### Future Work
1. Address remaining class imbalance (SMOTE, different weights)
2. Ensemble methods (RF + XGBoost)
3. Deep learning with attention on spatial features
4. Hyperparameter optimization

---
*Analysis completed: 2025-11-19*
*Branch: feature/data-leakage-analysis*
*Clean features: 36 (removed 7 leaked)*
*Shot percentage: 29.0% (including blocked shots)*