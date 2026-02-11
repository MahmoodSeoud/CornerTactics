# Phase 3: Model Architecture & Training

## Status: Complete

All tests passing (64 tests for Phase 3).

## Goal
Build ST-GNN, pretrain on open play, fine-tune on corners, run velocity ablation experiment.

## Implemented Modules

### model.py (232 lines)

**SpatialGNN**
- GATv2Conv layers to process single frame's graph
- Input: 8 node features per player/ball
- Output: 32-dim graph-level embedding via mean pooling
- Architecture: 2 GATv2Conv layers with LayerNorm and ReLU

**TemporalAggregator**
- 2-layer GRU to aggregate frame-level representations
- Input: sequence of 32-dim embeddings
- Output: 64-dim temporal embedding (final hidden state)

**CornerKickPredictor**
- Full ST-GNN combining SpatialGNN + TemporalAggregator
- Multi-head outputs:
  - head_shot: Binary shot prediction (sigmoid)
  - head_goal: Binary goal prediction (sigmoid)
  - head_contact: First contact team (2-class)
  - head_outcome: Outcome class (6-class)

### train.py (330 lines)

**extract_open_play_sequences(tracking, events, window_seconds, stride_seconds)**
- Extract overlapping windows for pretraining
- Labels each window: shot within 6 seconds?

**pretrain_spatial_gnn(model, sequences, epochs, lr)**
- Pretrain spatial GNN on open-play shot prediction
- Uses temporary head matching gnn_out dimension
- Trains only spatial_gnn weights

**leave_one_match_out_split(dataset)**
- Cross-validation splits without data leakage
- Returns list of (train, test) tuples

**compute_multi_task_loss(predictions, labels)**
- Combined loss for all prediction heads
- Weighted sum: shot + goal + 0.5*contact + 0.5*outcome

**finetune_on_corners(model, dataset, epochs)**
- Fine-tune full model on corner kicks
- Uses leave-one-match-out CV
- Returns per-fold results with predictions/labels

**zero_out_velocity_features(dataset)**
- Creates deep copy with vx=0, vy=0
- For position-only ablation condition

**run_ablation(dataset, epochs, lr)**
- Runs full ablation experiment
- Returns position_only and position_velocity results

### evaluate.py (197 lines)

**compute_auc(y_true, y_pred)**
- ROC AUC calculation using sklearn

**compute_f1(y_true, y_pred)**
- F1 score for binary classification

**aggregate_fold_results(fold_results)**
- Compute per-fold AUC and aggregate statistics
- Returns mean_auc, std_auc, fold_aucs

**paired_t_test(scores_a, scores_b)**
- Statistical significance testing
- Returns t_stat, p_value

**analyze_ablation_results(ablation_results)**
- Compare position-only vs position+velocity
- Computes delta AUC and p-value

**format_ablation_report(analysis)**
- Human-readable summary of ablation experiment

## Test Coverage

### test_model.py (25 tests)
- SpatialGNN: exists, dimensions, forward shapes, GATv2Conv
- TemporalAggregator: exists, dimensions, variable length
- CornerKickPredictor: components, forward, multi-head outputs

### test_train.py (20 tests)
- OpenPlayExtraction: returns list, required keys, window size
- PretrainSpatialGNN: returns model, modifies weights
- FinetuneOnCorners: returns results, fold metrics
- CrossValidation: no leakage, correct folds
- MultiTaskLoss: returns tensor, positive
- AblationExperiment: modifies dataset, deep copy

### test_evaluate.py (19 tests)
- Metrics: AUC, F1 computation
- CrossValidation: aggregate results
- StatisticalTests: paired t-test
- AblationAnalysis: key metrics, significance
- Formatting: report generation

## Usage Example

```python
from src.dfl import (
    load_tracking_data, load_event_data, build_corner_dataset_from_match,
    CornerKickPredictor, extract_open_play_sequences, pretrain_spatial_gnn,
    finetune_on_corners, run_ablation, analyze_ablation_results, format_ablation_report
)

# Load data
tracking = load_tracking_data("dfl", data_dir, match_id)
events = load_event_data("dfl", data_dir, match_id)

# Build corner dataset
dataset = build_corner_dataset_from_match(tracking, events, match_id)

# Create model
model = CornerKickPredictor()

# Pretrain on open play (optional but recommended)
open_play = extract_open_play_sequences(tracking, events)
model = pretrain_spatial_gnn(model, open_play, epochs=50)

# Run ablation experiment
ablation_results = run_ablation(dataset, epochs=100)
analysis = analyze_ablation_results(ablation_results)
print(format_ablation_report(analysis))
```

## Dependencies
- torch>=2.0
- torch-geometric>=2.4
- scipy
- scikit-learn
- numpy

## Expected Results
Following the thesis hypothesis:
- Position-only AUC: ~0.50 (random)
- Position+Velocity AUC: >0.55
- p-value < 0.10 for statistical significance
