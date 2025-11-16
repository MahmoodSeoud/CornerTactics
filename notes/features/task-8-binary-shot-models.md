# Task 8: Binary Shot Classification Models

## Goal
Train binary classifiers to predict whether a corner kick leads to a shot within the next 5 events.

## Status
✅ COMPLETE (with TacticAI-style filtering)

- Branch: task-8-binary-shot-models
- Task 7 Complete: Shot labels extracted with proper filtering
- Data: 1,933 corners with shot labels
- Class distribution: Shot 21.8% (422), No Shot 78.2% (1,511)
- Imbalance ratio: 3.58:1

## TacticAI Filtering Applied
Following the TacticAI paper methodology:
- ✅ Only count shots from **attacking team** (corner-taking team)
- ✅ Only count **threatening shots**: Goal, Saved, Post, Off Target, Wayward
- ✅ Exclude: Blocked shots, shots from defending team
- ✅ Lookahead window: 5 events

Result: 21.8% shot rate (close to TacticAI's reported ~24%)

## Results Summary

### Test Set Performance (407 samples, 84 shots)

**Random Forest** (BEST MODEL):
- Accuracy: 70.8%
- Precision (Shot): 34.2%
- Recall (Shot): 45.2%
- F1 (Shot): 0.390
- ROC-AUC: 0.638
- PR-AUC: 0.316

**XGBoost**:
- Accuracy: 68.3%
- Precision (Shot): 27.7%
- Recall (Shot): 33.3%
- F1 (Shot): 0.303
- ROC-AUC: 0.634
- PR-AUC: 0.294

**MLP**:
- Accuracy: 72.7%
- Precision (Shot): 32.0%
- Recall (Shot): 28.6%
- F1 (Shot): 0.302
- ROC-AUC: 0.633
- PR-AUC: 0.277

### Analysis
- Random Forest achieved best F1 score for shot class (0.390)
- MLP achieved highest overall accuracy (72.7%) but lower shot recall
- All models show reasonable performance for this challenging imbalanced task
- Shot percentage (21.8%) now matches TacticAI's methodology (~24%)
- More challenging task than initial implementation due to stricter filtering

## Notes
- Shot percentage (29.0%) is higher than expected (10-20% from literature)
- This may indicate lookahead window of 5 events is appropriate for this dataset
- Will use same features from Task 3 (corners_with_features.csv)
- Will use same train/val/test splits from Task 4

## Implementation Plan (TDD)
1. RED: Write test for data loading and merging shot labels with features
2. GREEN: Implement data loading function
3. REFACTOR: Clean up code
4. RED: Write test for Random Forest binary classifier
5. GREEN: Implement Random Forest with class_weight='balanced'
6. REFACTOR: Extract common training/evaluation code
7. RED: Write test for XGBoost binary classifier
8. GREEN: Implement XGBoost with scale_pos_weight
9. REFACTOR: Clean up
10. RED: Write test for MLP binary classifier
11. GREEN: Implement MLP with class weighting
12. REFACTOR: Final cleanup
13. RED: Write test for evaluation metrics (ROC-AUC, PR curve)
14. GREEN: Implement comprehensive evaluation
15. REFACTOR: Final polish

## Expected Performance
- Naive baseline: ~71% accuracy (always predict No Shot)
- Random Forest: 70-75% accuracy, F1 ~0.35-0.45 for Shot class
- XGBoost: 72-77% accuracy, F1 ~0.40-0.50 for Shot class
- MLP: 68-73% accuracy, F1 ~0.30-0.40 for Shot class
