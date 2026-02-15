# Task 4: Training & Evaluation — Implementation Plan

## Overview

Implement the complete training, evaluation, permutation testing, and ablation pipeline for the two-stage corner kick prediction model. All experiments submitted as SLURM jobs.

## Files to Create

```
corner_prediction/
├── config.py                           # Step 1: Hyperparameters & paths
├── training/
│   ├── __init__.py                     # Step 2: Package init
│   ├── train.py                        # Step 3: Core training functions
│   ├── evaluate.py                     # Step 4: LOMO CV harness + metrics
│   ├── permutation_test.py             # Step 5: Statistical validation
│   └── ablation.py                     # Step 6: Ablation runner
├── run_all.py                          # Step 7: Single entry point
scripts/slurm/
└── run_corner_prediction.sbatch        # Step 8: SLURM job script
tests/corner_prediction/
└── test_training.py                    # Step 9: Tests
```

---

## Step 1: `corner_prediction/config.py`

Centralized configuration. All hyperparameters, paths, and constants in one place.

```python
# Contents:
MATCH_IDS = ["2017461", "2015213", "2013725", "2011166", "2006229",
             "1996435", "1953632", "1925299", "1899585", "1886347"]

# Paths
DATA_DIR = "corner_prediction/data"
PRETRAINED_PATH = "transfer_learning/weights/ussf_backbone_dense.pt"
RESULTS_DIR = "results/corner_prediction"

# Training hyperparameters
RECEIVER_LR = 1e-3
RECEIVER_EPOCHS = 100
RECEIVER_PATIENCE = 20
RECEIVER_WEIGHT_DECAY = 1e-3
RECEIVER_HIDDEN = 64
RECEIVER_DROPOUT = 0.3

SHOT_LR = 1e-3
SHOT_EPOCHS = 100
SHOT_PATIENCE = 20
SHOT_WEIGHT_DECAY = 1e-3
SHOT_HIDDEN = 32
SHOT_DROPOUT = 0.3

BATCH_SIZE = 8  # small due to 86 samples

# Ablation configs (6 from pipeline doc)
ABLATION_CONFIGS = {
    "position_only": {...},    # x, y, team, role
    "plus_velocity": {...},    # + vx, vy, speed
    "plus_detection": {...},   # + is_detected
    "full_features": {...},    # all 13 features
    "full_fc_edges": {...},    # all 13 + fully connected edges
    "xgboost_baseline": {...}, # no GNN
}

SEEDS = [42, 123, 456, 789, 1234]
N_PERMUTATIONS = 100
```

---

## Step 2: `corner_prediction/training/__init__.py`

Exports for the training package.

---

## Step 3: `corner_prediction/training/train.py`

Core training functions for both stages.

### Functions:

**`train_receiver_epoch(model, loader, optimizer, device)`**
- model.predict_receiver() → logits → receiver_loss()
- Only graphs with has_receiver_label=True contribute
- Returns epoch loss

**`train_shot_epoch(model, loader, optimizer, device, pos_weight, receiver_mode)`**
- receiver_mode ∈ {"oracle", "predicted", "none"}
- "oracle": use ground-truth receiver_label as receiver_indicator
- "predicted": use model.predict_receiver() → argmax → receiver_indicator (detached)
- "none": no receiver indicator (unconditional baseline)
- model.predict_shot() → BCEWithLogitsLoss with pos_weight
- Returns epoch loss

**`eval_receiver(model, loader, device)`**
- Predict receivers across all graphs
- Returns: top1_acc, top3_acc, per_graph_probs, labels

**`eval_shot(model, loader, device, receiver_mode)`**
- Predict shots with given receiver conditioning
- Returns: AUC, F1, accuracy, probs, labels

**`build_model(backbone_mode, pretrained_path, freeze)`**
- Construct CornerBackbone + ReceiverHead + ShotHead + TwoStageModel
- Load pretrained if path exists and mode="pretrained"
- Return model

**`get_optimizer(model, stage, lr, weight_decay)`**
- Stage 1: optimize receiver_head params + backbone projection params
- Stage 2: optimize shot_head params + backbone projection params
- Filter requires_grad=True

**Key design decisions:**
- Two-phase training: train receiver head first (Stage 1), then freeze it and train shot head (Stage 2)
- Early stopping based on validation loss, with patience
- For Stage 2 training, use oracle receiver during training (more stable gradient signal) but evaluate with predicted receiver

---

## Step 4: `corner_prediction/training/evaluate.py`

LOMO cross-validation harness.

### Functions:

**`lomo_cv(dataset, config, seed=42)`**
Main cross-validation function:
```
for each of 10 matches as held_out:
    test = graphs from held_out match
    remaining = all other graphs
    val = graphs from one remaining match (rotate or select)
    train = rest

    model = build_model(config)

    # Phase 1: Train receiver head
    train Stage 1 with early stopping on val receiver loss

    # Phase 2: Train shot head (receiver head frozen)
    train Stage 2 with early stopping on val shot loss

    # Evaluate
    receiver_metrics = eval_receiver(model, test)
    shot_metrics_oracle = eval_shot(model, test, "oracle")
    shot_metrics_predicted = eval_shot(model, test, "predicted")
    shot_metrics_unconditional = eval_shot(model, test, "none")

    store per-fold results

return aggregated metrics (mean ± std across 10 folds)
```

Inner validation: For each LOMO fold, use the next match in sorted order as validation.
E.g., if held_out = match[0], val = match[1], train = match[2:9].

**`compute_receiver_metrics(probs, labels, masks, batch)`**
- Top-1 accuracy: argmax of probs among masked nodes == true receiver
- Top-3 accuracy: true receiver in top-3 by probability
- Per-fold, per-graph computation, then average

**`compute_shot_metrics(probs, labels)`**
- AUC-ROC (handle single-class folds by returning 0.5)
- F1 at optimal threshold
- Precision, recall

**`print_results_table(results)`**
- Formatted table for thesis: condition × metric with mean ± std

---

## Step 5: `corner_prediction/training/permutation_test.py`

Statistical validation via label shuffling.

### Functions:

**`permutation_test_receiver(dataset, config, n_permutations=100, seed=42)`**
```
real_metric = lomo_cv(dataset, config).receiver_top3_acc_mean
shuffled_metrics = []
for i in range(n_permutations):
    shuffle receiver labels (preserve mask structure)
    shuffled_metric = lomo_cv(shuffled_dataset, config).receiver_top3_acc_mean
    shuffled_metrics.append(shuffled_metric)
p_value = (sum(s >= real_metric for s in shuffled_metrics) + 1) / (n_permutations + 1)
```

**`permutation_test_shot(dataset, config, n_permutations=100, seed=42)`**
Same for shot AUC.

**`shuffle_receiver_labels(dataset)`**
- For each graph: randomly reassign receiver_label to one of the masked nodes
- Preserves mask structure and label rate

**`shuffle_shot_labels(dataset)`**
- Randomly permute shot_label across all graphs
- Preserves positive rate

Note: Full LOMO per permutation × 100 permutations × 10 folds = 1000 training runs.
With ~80 samples and fast training (~100 epochs), this should be feasible.
If too slow: use 1 seed instead of 5, or N=50 permutations.

---

## Step 6: `corner_prediction/training/ablation.py`

Run all ablation configurations from pipeline doc Table.

### Ablation configs:

| ID | Name | Node Features | Edge Type | Description |
|----|------|--------------|-----------|-------------|
| 0 | position_only | x, y, is_atk, is_taker, is_gk, role (9 feat) | KNN k=6 | No velocity |
| 1 | plus_velocity | x, y, vx, vy, speed, is_atk, is_taker, is_gk, role (13 feat) | KNN k=6 | Add velocity |
| 2 | plus_detection | all 13 + is_detected | KNN k=6 | Full features (same as default) |
| 3 | full_features | All 14 (13 + receiver indicator) | KNN k=6 | Full model |
| 4 | full_fc_edges | All 14 | Dense | FC edges |
| 5 | no_gnn | Aggregate features | N/A | XGBoost baseline |

Wait — re-reading the pipeline doc carefully:

| Experiment | Node features | Edge type | Notes |
|-----------|--------------|-----------|-------|
| Position only | x, y, team, role | KNN k=6 | Replicates 7.5 ECTS finding |
| + Velocity | x, y, vx, vy, speed, team, role | KNN k=6 | Tests velocity hypothesis |
| + Detection flag | Above + is_detected | KNN k=6 | Tests quality awareness |
| Full features | All 13 features | KNN k=6 | Full model |
| Full + FC edges | All 13 features | Fully connected | Edge construction impact |
| No GNN (baseline) | Aggregate features | N/A (XGBoost) | Traditional ML comparison |

Implementation: Each ablation config specifies which feature indices to zero out (mask approach)
or which features to include (subset approach).

**Mask approach**: Simpler — keep all 13 features but zero out excluded ones.
This way the model architecture stays the same (14-dim input to backbone).
The backbone still expects 14 features (13 + receiver indicator).

For "position_only": zero out vx, vy, speed, is_detected → keep [0,1,5,6,7,9,10,11,12]
For "+ velocity": zero out is_detected → keep all except [8]
For "+ detection": keep all 13 → same as "full features"...

Actually re-reading: "+ detection" adds is_detected on top of "+ velocity". The pipeline doc lists these as separate increments. Let me reparse:

- position_only: indices [0,1,5,6,7,9,10,11,12] = x, y, is_atk, is_taker, is_gk, role_onehot
- plus_velocity: position_only + [2,3,4] = add vx, vy, speed
- plus_detection: plus_velocity + [8] = add is_detected (= all 13)
- full_features: all 13 (same as plus_detection actually)

Hmm, "plus_detection" and "full_features" would be identical. Let me re-interpret:

- position_only = [x, y, is_atk, is_gk, role] (no is_taker, no velocity, no detected)
  Actually the doc says "x, y, team, role" so: [0,1,5,9,10,11,12] = 7 features
- plus_velocity = position_only + [vx, vy, speed] = [0,1,2,3,4,5,9,10,11,12] = 10 features
- plus_detection = + [is_detected] = 11 features
- full_features = + [is_corner_taker, is_goalkeeper as separate] = all 13

Actually it's cleaner to implement as a feature mask: each config specifies which of the 13 input features are active. Inactive features get zeroed.

For the XGBoost baseline: aggregate graph features into a flat vector (mean/max of player features by team). This is a separate code path.

### Functions:

**`run_ablation(ablation_id, dataset, seed=42)`**
- Load config for this ablation
- Apply feature masking to dataset
- Run lomo_cv()
- Return results

**`run_all_ablations(dataset, seeds)`**
- Run each ablation with LOMO CV
- Return combined results table

**`apply_feature_mask(dataset, mask_indices)`**
- Zero out features not in mask_indices for all graphs in dataset
- Return modified dataset (deep copy)

---

## Step 7: `corner_prediction/run_all.py`

Single entry point with CLI.

```bash
# Full pipeline
python -m corner_prediction.run_all

# Just LOMO evaluation
python -m corner_prediction.run_all --eval-only

# Just permutation test
python -m corner_prediction.run_all --permutation-only --n-permutations 100

# Just one ablation
python -m corner_prediction.run_all --ablation position_only

# All ablations
python -m corner_prediction.run_all --all-ablations
```

### CLI args:
- `--mode`: "pretrained" or "scratch"
- `--seed / --seeds`
- `--eval-only`: skip permutation/ablation
- `--permutation-only`: only run permutation tests
- `--all-ablations`: run all ablation configs
- `--ablation <name>`: run single ablation
- `--n-permutations`: default 100
- `--output-dir`: results directory
- `--no-gpu`: force CPU

### Output:
- `results/corner_prediction/lomo_results_{timestamp}.pkl` — full LOMO results
- `results/corner_prediction/lomo_summary_{timestamp}.json` — formatted summary
- `results/corner_prediction/permutation_{timestamp}.pkl` — permutation test results
- `results/corner_prediction/ablation_{name}_{timestamp}.pkl` — per-ablation results

---

## Step 8: `scripts/slurm/run_corner_prediction.sbatch`

SLURM array job script.

```bash
#SBATCH --job-name=corner_pred
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/corner_pred_%A_%a.out
#SBATCH --error=logs/corner_pred_%A_%a.err

# Array tasks:
#   0 = Main LOMO evaluation (pretrained backbone)
#   1 = Main LOMO evaluation (scratch backbone)
#   2 = Ablation: position_only
#   3 = Ablation: plus_velocity
#   4 = Ablation: plus_detection
#   5 = Ablation: full_fc_edges
#   6 = Permutation test (receiver)
#   7 = Permutation test (shot)
```

---

## Step 9: `tests/corner_prediction/test_training.py`

Tests for training pipeline correctness.

### Test cases:

**TestTrainReceiverEpoch:**
- One epoch reduces loss (or at least doesn't crash)
- Only labeled graphs contribute to loss
- Gradients flow to receiver_head and projection, not frozen backbone

**TestTrainShotEpoch:**
- One epoch with oracle receiver
- One epoch with predicted receiver
- One epoch unconditional
- pos_weight changes loss magnitude

**TestEvalReceiver:**
- Top-1 accuracy computation correct
- Top-3 accuracy computation correct
- Handles single-graph batches

**TestEvalShot:**
- AUC computation correct
- Handles all-same-label edge case (returns 0.5)

**TestLOMOCV:**
- 10 folds produced
- No data leakage (test match not in train)
- Inner val split works

**TestFeatureMask:**
- Zeroed features are actually zero
- Non-zeroed features unchanged
- Mask doesn't modify original dataset

All tests use synthetic data (small graphs with known properties), not real data files.

---

## Execution Order

1. config.py — foundations, no dependencies
2. training/__init__.py — empty package init
3. training/train.py — core training logic
4. training/evaluate.py — LOMO harness, depends on train.py
5. training/permutation_test.py — depends on evaluate.py
6. training/ablation.py — depends on evaluate.py
7. run_all.py — entry point, depends on all above
8. SLURM script — references run_all.py
9. Tests — verify all above

## Test Criteria

- All tests pass with synthetic data
- `python -m corner_prediction.run_all --eval-only` completes on real data
- SLURM script submits without error
- Results pickle files are loadable and contain expected keys
