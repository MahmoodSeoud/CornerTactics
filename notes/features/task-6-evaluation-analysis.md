# Task 6: Evaluation & Analysis

## Overview
Create `scripts/06_evaluate_models.py` to comprehensively evaluate trained baseline models.

## Requirements
1. Load all trained models (RF, XGBoost, MLP)
2. Generate predictions on test set
3. Create visualizations:
   - Confusion matrices (3 plots, one per model)
   - Feature importance plot (for RF and XGBoost)
   - Per-class F1 comparison (bar chart)
4. Error analysis:
   - Find most confused pairs
   - Analyze feature distributions for misclassified samples
5. Save report to `results/evaluation_report.md`

## Prerequisites
- Task 5 must be complete (models trained)
- Test data available in `data/processed/`
- Trained models in `models/` directory

## Expected Outputs
- `results/confusion_matrices/` directory with 3 PNG files
- `results/feature_importance.png`
- `results/evaluation_report.md`

## TDD Approach
1. Test: Model loading functionality
2. Test: Confusion matrix generation
3. Test: Feature importance extraction
4. Test: Error analysis logic
5. Test: Report generation

## Implementation Status: COMPLETE ✓

### Test Results
- All 12 tests passing
- Test coverage includes model loading, confusion matrices, feature importance, error analysis, and report generation
- Integration test validates full pipeline

### Generated Outputs
- `results/confusion_matrices/random_forest_confusion_matrix.png`
- `results/confusion_matrices/xgboost_confusion_matrix.png`
- `results/confusion_matrices/mlp_confusion_matrix.png`
- `results/feature_importance.png`
- `results/per_class_f1.png`
- `results/evaluation_report.md`

### Key Findings from Evaluation
**Test Set Size**: 407 samples

**Model Performance**:
- Random Forest: 51% accuracy, macro F1: 0.43
- XGBoost: 50% accuracy, macro F1: 0.40
- MLP: 50% accuracy, macro F1: 0.20

**Most Important Features** (Random Forest):
1. pass_length: 0.1661
2. pass_end_x: 0.1584
3. pass_end_y: 0.1138
4. defending_depth: 0.0512
5. defending_to_goal_dist: 0.0467

**Most Confused Pairs** (Random Forest):
- Clearance → Ball Receipt: 45 times
- Ball Receipt → Clearance: 42 times
- Ball Receipt → Goalkeeper: 32 times

### Refactoring
Fixed bug where feature importance was not being extracted correctly in the report generation due to model wrapping for scaling. Unwrapped models before feature importance extraction.

### Notes
- MLP struggles with minority classes (Goalkeeper, Other) - predicts 0 instances
- Tree models (RF, XGBoost) perform better with class balance
- Pass trajectory features are most important for prediction
- Confusion is highest between Ball Receipt and Clearance (defensive actions)
