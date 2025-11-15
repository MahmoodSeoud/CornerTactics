# Training Plan for 34K Corner Kick Dataset (Clean Slate)

## Context
You have a dataset of 34,049 corner kicks with event sequence data. You will predict which of the 15 default event types (from the transition bar chart) will occur after a corner kick. Start fresh - no pre-engineered features, just the raw data StatsBomb provides.

## Phase 1: Data Preparation & Baseline Training

### Task 1.1: Load and explore the 34K dataset
- Load the corner kick dataset (34,049 instances) with its raw features that we have.
- Identify what raw features are available in the StatsBomb data by default (DO NOT engineer new ones)
- Possible raw features might include:
  - Player positions (x, y coordinates)
  - Team identities
  - Corner kick location
  - Timestamp
  - Player IDs
- Document exactly what features exist in the raw data

### Task 1.2: Define prediction tasks

**Task 1: Receiver Prediction**
- Target: Which player receives the ball first?
- Metric: Top-1, Top-3, Top-5 accuracy

**Task 2: Event Outcome Prediction**
- Target: Which of the 15 event types from the transition bar chart occurs?
  - Ball Receipt (57.8%)
  - Clearance (22.9%)
  - Goal Keeper (7.6%)
  - Duel (Aerial Lost): Unknown (3.4%)
  - Pressure (2.9%)
  - Foul Committed (1.3%)
  - Ball Recovery (1.0%)
  - Block (0.6%)
  - Plus 7 more rare events
- Metrics: Accuracy, Macro F1, Per-class F1 scores
- (see data/analysis/corner_transitions_report.md for full event types)

### Task 1.3: Create train/val/test splits
- Split: 70/15/15
- Document class distributions
- Check for severe class imbalance (use techniques like class weights if needed)

### Task 1.4: Hyperparameter search for architectures

**Model A: Random Baseline**
- No hyperparameters needed
- Just establish the floor

**Model B: MLP with hyperparameter search**
- Search space:
  - Architecture:
    - Depth: [2, 3, 4, 5] hidden layers
    - Width: [64, 128, 256, 512] neurons per layer
    - Try combinations like:
      - [256, 128]
      - [512, 256, 128]
      - [256, 256, 256]
      - [512, 256, 128, 64]
  - Learning rate: [0.0001, 0.001, 0.01]
  - Batch size: [16, 32, 64, 128]
  - Dropout: [0.0, 0.1, 0.2, 0.3, 0.5]
  - Optimizer: [Adam, SGD, AdamW]
  - Weight decay: [0, 1e-5, 1e-4]
- Use validation set to select best hyperparameters
- Document final chosen architecture and why

**Model C: XGBoost with hyperparameter search**
- Search space:
  - max_depth: [3, 5, 7, 10]
  - n_estimators: [50, 100, 200, 300, 500]
  - learning_rate: [0.01, 0.05, 0.1, 0.3]
  - subsample: [0.6, 0.8, 1.0]
  - colsample_bytree: [0.6, 0.8, 1.0]
  - min_child_weight: [1, 3, 5]
  - gamma: [0, 0.1, 0.2]
- Use validation set to select best hyperparameters
- Document final chosen parameters and why

### Task 1.5: Track overfitting during training

For both MLP and XGBoost, save CSV/JSON data for:

**Overfitting diagnostics:**
1. **Learning curves** (CSV):
   - Columns: epoch, train_loss, val_loss
   - For plotting: training loss vs validation loss over epochs/iterations

2. **Performance curves** (CSV):
   - Columns: epoch, train_accuracy, val_accuracy, train_f1, val_f1
   - For plotting: training vs validation metrics

3. **Train/Val/Test comparison** (JSON):
   - Save final metrics for each model:
   ```json
   {
     "MLP": {
       "train_acc": 0.XX, "val_acc": 0.XX, "test_acc": 0.XX,
       "train_f1": 0.XX, "val_f1": 0.XX, "test_f1": 0.XX
     },
     "XGBoost": { ... }
   }
   ```

4. **Overfitting metrics** (CSV):
   - Columns: model, train_val_gap, relative_degradation, severe_overfitting
   - Train-Val gap: |Train_Acc - Val_Acc|
   - Relative degradation: (Train_Acc - Val_Acc) / Train_Acc × 100%
   - Flag if degradation > 10% (severe overfitting)

### Task 1.6: Evaluate final models

Save results as CSV/JSON:

**CSV 1: Task 1 - Receiver Prediction (on test set)**
- File: `results/task1_receiver_prediction.csv`
- Columns: model, top1_accuracy, top3_accuracy, top5_accuracy
- Rows: Random, MLP, XGBoost

**CSV 2: Task 2 - Event Outcome Classification (15 classes)**
- File: `results/task2_outcome_classification.csv`
- Columns: model, accuracy, macro_f1
- Rows: Random, MLP, XGBoost

**CSV 3: Per-class F1 scores**
- File: `results/task2_per_class_f1.csv`
- Columns: model, event_type, f1_score, event_frequency
- Include all 15 event types, minimum:
  - Ball Receipt (57.8%)
  - Clearance (22.9%)
  - Goal Keeper (7.6%)
  - Duel (Aerial Lost): Unknown (3.4%)
  - Pressure (2.9%)
  - etc.

**JSON: Confusion matrices**
- File: `results/task2_confusion_matrices.json`
- Format: `{"MLP": [[...]], "XGBoost": [[...]]}`
- 15×15 matrices for best models

---

## Phase 2: Feature Importance Analysis

### Task 2.1: XGBoost feature importance
- Extract feature importance using XGBoost's `gain` metric
- Save as CSV:
  - File: `results/feature_importance.csv`
  - Columns: feature_name, importance_gain, importance_percentage, rank
  - Sort by importance descending
  - Include all features

### Task 2.2: Feature correlation with outcomes
- Compute Pearson correlation between each feature and the 15 outcome classes
- Save as CSV:
  - File: `results/feature_outcome_correlations.csv`
  - Columns: feature_name, event_type, correlation, abs_correlation, p_value
  - Include all feature × event combinations
- Summary JSON:
  - File: `results/correlation_summary.json`
  - Include: max_correlation, features_with_signal (|r| > 0.1), strong_correlations (|r| > 0.3)

### Task 2.3: Document why prediction is hard
- Save analysis as JSON:
  - File: `results/prediction_difficulty_analysis.json`
  - Include:
    - `max_correlation`: highest |r| value found
    - `weak_correlations`: count of features with |r| < 0.15
    - `strong_correlations`: count of features with |r| > 0.3
    - `interpretation`: text explaining why prediction is hard/easy based on correlations

---

## Phase 3: Ablation Study - Feature Selection

### Task 3.1: Identify feature importance tiers
Based on XGBoost feature importance:
- **Tier 1**: Features with >10% total importance
- **Tier 2**: Features with 5-10% importance
- **Tier 3**: Features with 1-5% importance
- **Tier 4**: Features with <1% importance (candidates for removal)

### Task 3.2: Systematic ablation experiments

**Config A: Full features (baseline)**
- Use all raw features available
- This is your baseline

**Config B: Remove Tier 4 features (bottom <1% importance)**
- Remove all features contributing <1% importance
- Retrain XGBoost and MLP
- Compare to baseline

**Config C: Keep only Tier 1+2 (top features >5% importance)**
- Remove all features below 5% importance
- Retrain models
- Compare to baseline

**Config D: Keep only Tier 1 (top features >10% importance)**
- Most aggressive pruning
- Retrain models
- Compare to baseline

**Config E: Keep only top 10 features**
- Regardless of importance tier, keep exactly top 10
- Retrain models
- Compare to baseline

### Task 3.3: Ablation results
- Save as CSV:
  - File: `results/ablation_results.csv`
  - Columns: config, num_features, task1_top3, task2_acc, task2_f1, delta_acc, delta_f1
  - Rows: Full (baseline), Remove Tier 4, Keep Tier 1+2, Keep Tier 1 only, Top 10 only

### Task 3.4: Statistical significance
- Perform McNemar's test or paired t-test comparing full features vs ablated configs
- Save as CSV:
  - File: `results/ablation_statistical_tests.csv`
  - Columns: config_comparison, test_type, p_value, significant (p<0.05)

### Task 3.5: Overfitting analysis per config
- Track train/val/test performance for each ablation config
- Save as CSV:
  - File: `results/ablation_overfitting.csv`
  - Columns: config, train_acc, val_acc, test_acc, overfitting_gap, train_f1, val_f1, test_f1

---

## Phase 4: Analysis & Documentation

### Task 4.1: Key findings summary

1. **Dataset scale impact**
   - With 34K corners (vs previous work with 7K):
     - What performance did we achieve?
     - Compare to TacticAI's 78% Top-3 receiver accuracy (they had 7,176 corners)

2. **Model comparison**
   - Does XGBoost outperform MLP?
   - Which model overfits more?
   - Which is more sample-efficient?

3. **Feature importance insights**
   - Which raw features from StatsBomb matter most?
   - What percentage of importance do top 5 features capture?
   - Are most features redundant?

4. **Outcome prediction difficulty**
   - Is predicting 15 event types harder than 3-class outcome?
   - Which events are easiest/hardest to predict?
   - Does class imbalance hurt rare event prediction?

5. **Ablation insights**
   - Can we achieve 95%+ of baseline performance with 50% fewer features?
   - Does feature pruning reduce overfitting?
   - Identify minimal feature set for deployment

### Task 4.2: Verify all CSV/JSON files are saved

**Data files needed for figures (already saved in previous tasks):**
1. **Learning curves**: `results/learning_curves_mlp.csv`, `results/learning_curves_xgboost.csv`
2. **Feature importance**: `results/feature_importance.csv`
3. **Correlation heatmap**: `results/feature_outcome_correlations.csv`
4. **Ablation comparison**: `results/ablation_results.csv`
5. **Confusion matrices**: `results/task2_confusion_matrices.json`
6. **Overfitting comparison**: `results/overfitting_metrics.csv`, `results/train_val_test_comparison.json`
7. **Per-class F1 scores**: `results/task2_per_class_f1.csv`

### Task 4.3: Verify all CSV/JSON files for LaTeX tables

**Data files needed for tables (already saved in previous tasks):**
- Baseline results: `results/task1_receiver_prediction.csv`, `results/task2_outcome_classification.csv`
- Hyperparameter search: `models/hyperparameter_search/mlp_search_results.json`, `models/hyperparameter_search/xgboost_search_results.json`
- Overfitting comparison: `results/train_val_test_comparison.json`
- Feature importance top 10: `results/feature_importance.csv` (filter top 10)
- Ablation study: `results/ablation_results.csv`
- Statistical tests: `results/ablation_statistical_tests.csv`

---

## Deliverables Checklist

**Trained models:**
- [ ] `models/final/mlp_receiver_best.pth`
- [ ] `models/final/mlp_outcome_best.pth`
- [ ] `models/final/xgboost_receiver_best.pkl`
- [ ] `models/final/xgboost_outcome_best.pkl`

**CSV/JSON data files (NO figures, NO LaTeX):**
- [ ] `results/task1_receiver_prediction.csv`
- [ ] `results/task2_outcome_classification.csv`
- [ ] `results/task2_per_class_f1.csv`
- [ ] `results/task2_confusion_matrices.json`
- [ ] `results/learning_curves_mlp.csv`
- [ ] `results/learning_curves_xgboost.csv`
- [ ] `results/performance_curves_mlp.csv`
- [ ] `results/train_val_test_comparison.json`
- [ ] `results/overfitting_metrics.csv`
- [ ] `results/feature_importance.csv`
- [ ] `results/feature_outcome_correlations.csv`
- [ ] `results/correlation_summary.json`
- [ ] `results/prediction_difficulty_analysis.json`
- [ ] `results/ablation_results.csv`
- [ ] `results/ablation_statistical_tests.csv`
- [ ] `results/ablation_overfitting.csv`
- [ ] `models/hyperparameter_search/mlp_search_results.json`
- [ ] `models/hyperparameter_search/xgboost_search_results.json`

**Text summaries (markdown or txt):**
- [ ] Key findings (3-5 bullets per section)
- [ ] Hyperparameter justification (why we chose these values)
- [ ] Overfitting analysis interpretation
- [ ] Feature importance interpretation
- [ ] Ablation study conclusions

---

## Implementation Notes

**Output format requirements:**
- Save ALL results as CSV or JSON files
- DO NOT generate any PNG, PDF, or image files
- DO NOT generate LaTeX code or tables directly
- All data should be in machine-readable format for later visualization/table generation

**Reproducibility:**
- Set random seeds: 42 everywhere
- Document Python/library versions
- Save hyperparameter configs
- Save train/val/test indices as JSON

---

## Execution Order

1. **Phase 1.1-1.3**: Load data, explore, split
2. **Phase 1.4**: Hyperparameter search (this will take time!)
3. **Phase 1.5-1.6**: Train final models, track overfitting, evaluate (save as CSV/JSON)
4. **Phase 2**: Analyze feature importance and correlations (save as CSV/JSON)
5. **Phase 3**: Run ablation experiments (save as CSV/JSON)
6. **Phase 4**: Verify all CSV/JSON files exist and are properly formatted

---

## Key Questions to Answer

After running this plan, you should be able to answer:

1. **With 34K corners, what's the best performance achievable with raw features?**
2. **Do we overfit or underfit with this dataset size?**
3. **Which raw features actually matter for corner kick prediction?**
4. **Can we remove 50% of features without hurting performance?**
5. **Why is outcome prediction hard even with 34K examples?**
6. **How do we compare to TacticAI now that we have MORE data?**
