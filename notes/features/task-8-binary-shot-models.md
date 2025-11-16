# Task 8: Binary Shot Classification Models

## Goal
Train binary classifiers to predict whether a corner kick leads to a shot within the next 5 events.

## Status
- Branch: task-8-binary-shot-models
- Task 7 Complete: Shot labels extracted
- Data: 1,933 corners with shot labels
- Class distribution: Shot 29.0% (560), No Shot 71.0% (1,373)
- Imbalance ratio: 2.45:1

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
