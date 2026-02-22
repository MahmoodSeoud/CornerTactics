# Implementation Plan: USSF-Aligned Feature Engineering (Experiment A)

## Overview
Replace learned adapter layers (node_proj, edge_proj) with deterministic USSF-aligned feature engineering. Changes to 8 files, adds tests, preserves backward compatibility.

---

## Step 1: `corner_prediction/config.py` — Add USSF config

**Changes:**
- Add `FEATURE_MODE = "pretrained"` (default, backward-compatible)
- Add USSF constants: `PITCH_LENGTH`, `PITCH_WIDTH`, `SPEED_NORM`, `DIST_GOAL_NORM`, `DIST_BALL_NORM`, `GOAL_POS`
- Add `USSF_ABLATION_CONFIGS` with velocity ablation for 12-feature layout:
  - `ussf_position_only`: zero out [2,3,4,5] (velocity features)
  - `ussf_full`: all 12 active (baseline)

**Test criteria:** Import config, verify constants exist.

---

## Step 2: `corner_prediction/data/build_graphs.py` — Add USSF graph builder

**Changes — ADD new functions (do not modify existing):**

```python
# Constants (top of file)
USSF_NODE_DIM = 12
USSF_EDGE_DIM = 6

def build_ussf_node_features(player, ball_x, ball_y):
    """Build 12-dim USSF-aligned node feature vector."""
    # Returns: [px, py, vx_unit, vy_unit, vel_mag, vel_angle,
    #           dist_goal, angle_goal, dist_ball, angle_ball,
    #           attacking_team, potential_receiver(=0)]

def build_ussf_ball_features(ball_x, ball_y):
    """Build 12-dim features for ball node (index 22, last)."""

def build_ussf_edge_features(xi, yi, xj, yj, vel_i, vel_j, vx_i, vy_i, vx_j, vy_j):
    """Build 6-dim USSF-aligned edge features."""

def build_ussf_dense_edges_with_selfloops(n_nodes):
    """23×23 = 529 edges including self-loops."""

def corner_record_to_ussf_graph(record):
    """Convert corner record → PyG Data with 23 nodes, 12 features, 6 edge features."""
    # 22 player nodes + 1 ball node (last)
    # Dense edges with self-loops (529)
    # Self-loop features: [0, 0, 1.0, 0.5, 0.5, 0.5]
    # receiver_mask: 23 bools (ball node = False)
    # receiver_label: 23 floats (ball node = 0.0)

def build_ussf_graph_dataset(records):
    """Convert all records to USSF-aligned graphs."""
```

**Key decisions:**
- Player ordering: same as existing `corner_record_to_graph()` — players list order from record
- Ball node: appended LAST at index 22
- `receiver_mask[22] = False` (ball is not a receiver candidate)
- `receiver_label[22] = 0.0`
- `has_receiver_label`: unchanged from record
- Feature 11 (`potential_receiver`): always 0 at graph build time; modified at forward time by `TwoStageModel`

**Test criteria:**
- Graph from combined_corners.pkl has shape x=[23,12], edge_index=[2,529], edge_attr=[529,6]
- Ball node (22): dist_ball≈0, angle_ball≈0, attacking_team=0
- Self-loop edges have features [0,0,1.0,0.5,0.5,0.5]
- Feature ranges: x,y∈[0,1], vx,vy∈[-1,1], vel_mag∈[0,1], vel_angle∈[0,1]

---

## Step 3: `corner_prediction/data/dataset.py` — Route to USSF builder

**Changes:**
- Add `feature_mode` parameter to `CornerKickDataset.__init__()`
- In `process()`: if `feature_mode == "ussf_aligned"`, call `build_ussf_graph_dataset()` instead of `build_graph_dataset()`
- Update `processed_file_names` to include feature mode in the cache filename

**Test criteria:** Dataset loads with feature_mode="ussf_aligned", returns 143 graphs with correct shapes.

---

## Step 4: `corner_prediction/models/backbone.py` — Add ussf_aligned mode

**Changes:**
- Add `mode="ussf_aligned"` to `__init__`:
  ```python
  elif mode == "ussf_aligned":
      self._hidden_channels = hidden_channels or _USSF_HIDDEN  # 128
      self._init_ussf_aligned(num_conv_layers)
      if pretrained_path is not None:
          self.load_pretrained(pretrained_path)
      if freeze:
          self._freeze_backbone()
  ```
- `_init_ussf_aligned()`: same as pretrained but `node_proj=None`, `edge_proj=None`, `conv1(12, dim=6)`, `lin_in(12,128)`, `convs(128, dim=6) × 2`
- Update mode validation to accept "ussf_aligned"

**Forward path:** The existing forward already handles `node_proj is None` (skip projection). No changes needed there.

**Test criteria:**
- `CornerBackbone(mode="ussf_aligned")` creates with `node_proj=None`
- `output_dim == 128`
- Forward with [N, 12] nodes and [E, 6] edges produces [N, 128]
- Load pretrained weights works
- All backbone params frozen when freeze=True

---

## Step 5: `corner_prediction/models/two_stage.py` — Handle receiver indicator

**Changes to `_augment_with_receiver()`:**
- Add `ussf_mode` parameter (or check input dim)
- For ussf_aligned: clone x and set `x[:, 11] = receiver_indicator` (write to potential_receiver slot)
- For original: append as 14th column (current behavior)

**Changes to `predict_receiver()`, `predict_shot()`, `forward_two_stage()`:**
- Pass backbone mode to `_augment_with_receiver()` so it knows which path to take
- Alternative: check `x.shape[1]` — if 12, use USSF path; if 13, use original path

**Design choice:** Use `self.backbone.mode` check — explicit and clear:
```python
if self.backbone.mode == "ussf_aligned":
    x_aug = x.clone()
    if receiver_indicator is not None:
        x_aug[:, 11] = receiver_indicator
    # else feature 11 stays as-is (should be 0 from graph construction)
else:
    x_aug = self._augment_with_receiver(x, receiver_indicator)
```

**Test criteria:**
- USSF-aligned model with receiver indicator modifies feature 11 (not appending)
- Output shape unchanged [N, 128] embeddings
- Receiver conditioning changes shot prediction

---

## Step 6: `corner_prediction/training/train.py` — Update build_model

**Changes to `build_model()`:**
- Accept `node_features` and `edge_features` params (instead of hardcoded 14, 4)
- When `backbone_mode == "ussf_aligned"`: use `node_features=12`, `edge_features=6`
- Pass through to `CornerBackbone()`

**Changes to `train_fold()`:**
- The trainable_params loop filters for "node_proj"/"edge_proj" — these won't exist for ussf_aligned, so the loop naturally skips them. No change needed.
- The `_augment_with_receiver` calls in `train_receiver_epoch()` need to use the new logic from Step 5. Since they call `TwoStageModel._augment_with_receiver()` as a static method, we need to adjust.

**Detailed fix for `train_receiver_epoch()` (line 115) and `train_fold()` (line 486):**
These call `TwoStageModel._augment_with_receiver(batch.x)` directly. Must change to use model instance method that knows the mode. Options:
1. Pass model to augment call → `model._augment_for_mode(batch.x, None)`
2. Use the instance methods instead: call `model.predict_receiver()` which handles augmentation internally.

Looking more carefully: `train_receiver_epoch()` already exists and manually calls `_augment_with_receiver` + `backbone` + `receiver_head` separately. For ussf_aligned, the augmentation logic changes. Best approach: add `_prepare_features()` method to TwoStageModel that handles both modes, and use it everywhere.

**Test criteria:** `build_model(backbone_mode="ussf_aligned")` creates model with correct dims.

---

## Step 7: `corner_prediction/run_all.py` — Add --feature-mode CLI

**Changes:**
- Add `--feature-mode` argument: choices=["pretrained", "ussf_aligned"], default from config
- Thread `feature_mode` into `load_dataset()` call
- Thread `feature_mode` into `run_eval()`, `run_multi_seed()`, etc.
- When `feature_mode == "ussf_aligned"`, override `args.mode = "ussf_aligned"` for backbone
- Adjust save filename prefix to include "ussf_aligned" when applicable

**Test criteria:** `python -m corner_prediction.run_all --feature-mode ussf_aligned --help` works.

---

## Step 8: `corner_prediction/training/ablation.py` — USSF ablation configs

**Changes:**
- `apply_feature_mask()`: accept n_features parameter (13 or 12)
- Add USSF ablation support: when feature_mode is ussf_aligned, use 12-feature indices
- The ablation configs in config.py handle the feature index mapping

**Test criteria:** Velocity ablation zeros out indices [2,3,4,5] on 12-feature graphs.

---

## Step 9: `corner_prediction/baselines/mlp_baseline.py` — Dynamic dimensions

**Changes:**
- `_flatten_graph()`: detect `n_nodes` and `n_features` from `graph.x.shape` instead of hardcoding 22×13
- `FLAT_DIM`: compute dynamically from first graph (or accept parameter)
- `ShotMLP`/`ShotLinear`: accept `input_dim` parameter (already does)
- `_mlp_fold()`: compute `FLAT_DIM` from data

**Test criteria:** MLP works with 23×12=276 features and 22×13=286 features.

---

## Step 10: `corner_prediction/baselines/xgboost_baseline.py` — Adapt feature extraction

**Changes:**
- `extract_features()`: detect feature layout from `graph.x.shape[1]`
- For 12-feature (USSF): is_attacking at index 10 (not 5), positions at 0-1, velocity at 2-4
- For 13-feature (original): keep existing logic

**Test criteria:** XGBoost extracts 27 features from both layouts.

---

## Step 11: `tests/corner_prediction/test_models.py` — Add USSF tests

**Changes — ADD new test class `TestUSSFAligned`:**
- Test backbone creation with mode="ussf_aligned"
- Test forward pass with 23 nodes, 12 features, 6 edge features
- Test receiver indicator modifies feature 11
- Test end-to-end pipeline with USSF-aligned data
- Test `build_ussf_graph_dataset()` output shapes and ranges

---

## Step 12: Run existing tests

```bash
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate
python -m pytest tests/ -v
```

Verify all existing tests still pass (backward compatibility).

---

## Step 13: Run evaluations (SLURM or local)

```bash
# 5-seed evaluation
for seed in 42 123 456 789 1234; do
    python -m corner_prediction.run_all --eval --combined --seed $seed --feature-mode ussf_aligned
done

# Velocity ablation
python -m corner_prediction.run_all --ablation ussf_position_only --feature-mode ussf_aligned

# Permutation test
python -m corner_prediction.run_all --permutation-only --combined --seed 42 --feature-mode ussf_aligned

# Baselines
python -m corner_prediction.run_all --baselines all --combined --feature-mode ussf_aligned
```

---

## File Change Summary

| File | Type | Lines Changed (est.) |
|------|------|---------------------|
| config.py | MODIFY | +30 |
| data/build_graphs.py | MODIFY (add functions) | +180 |
| data/dataset.py | MODIFY | +15 |
| models/backbone.py | MODIFY | +30 |
| models/two_stage.py | MODIFY | +25 |
| training/train.py | MODIFY | +20 |
| run_all.py | MODIFY | +25 |
| training/ablation.py | MODIFY | +15 |
| baselines/mlp_baseline.py | MODIFY | +10 |
| baselines/xgboost_baseline.py | MODIFY | +30 |
| tests/test_models.py | MODIFY (add tests) | +80 |
| **Total** | | **~460 lines** |

## Execution Order

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13

Steps 1-2 are the foundation. Steps 3-7 wire it together. Steps 8-10 update baselines. Steps 11-12 verify. Step 13 runs experiments.

## Rules

- DO NOT modify evaluate.py or permutation_test.py
- DO NOT change hyperparameters
- DO NOT delete existing code
- Keep mode="pretrained" working
- All new result files include "ussf_aligned" in name
