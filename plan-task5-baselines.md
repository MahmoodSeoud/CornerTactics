# Task 5: Baseline Comparisons — Implementation Plan

## Overview

Implement 5 baseline models for contextualizing GNN two-stage corner kick prediction results. All baselines use the same LOMO (Leave-One-Match-Out) evaluation as the main model, producing results in the same format for direct comparison.

## Baselines

| # | Name | Training? | Receiver? | Shot? | Compute |
|---|------|-----------|-----------|-------|---------|
| 1 | Random | No | Yes (uniform) | Yes (dataset rate) | CPU |
| 2 | Heuristic Receiver | No | Yes (nearest to goal) | No | CPU |
| 3 | XGBoost Aggregate | Yes (sklearn) | No | Yes | CPU |
| 4 | Unconditional GNN | Already computed | No | Already in lomo_cv | GPU |
| 5 | MLP Flat | Yes (PyTorch) | No | Yes | CPU/GPU |

Baseline 4 requires no new code — `lomo_cv()` already produces `shot_unconditional`.

---

## Step 1: Create baselines package structure

### 1.1 Create `corner_prediction/baselines/__init__.py`
- Empty init or import public API

### 1.2 Create `corner_prediction/baselines/random_baseline.py`

**Function: `random_baseline_lomo(dataset, seed=42) -> Dict`**

Logic:
- For each LOMO fold (10 folds):
  - Split train/test using `lomo_split()`
  - For each test graph:
    - **Receiver**: Sample uniformly from `receiver_mask` candidates → compute top1/top3 (repeat N=100 times, take mean for stable estimate)
    - **Shot**: Compute dataset shot rate from train set. Predict all test samples with this probability. AUC = 0.5 by construction (constant predictor). Compute F1 at optimal threshold.
- Aggregate across folds in same format as `lomo_cv()` returns.

Returns: Same structure as `lomo_cv()` result — `per_fold`, `aggregated` with `receiver`, `shot_oracle` (= shot_unconditional since no receiver info).

### 1.3 Create `corner_prediction/baselines/heuristic_receiver.py`

**Function: `heuristic_receiver_lomo(dataset, seed=42) -> Dict`**

Logic:
- Goal center is at x=52.5, y=0.0 (corners are normalized: attacking left-to-right)
- For each LOMO fold:
  - For each test graph:
    - Get (x, y) from node features (indices 0, 1) — these are normalized; need to check if they need denormalization. Look at `build_graphs.py` to see normalization.
    - Filter to `receiver_mask == True` nodes (attacking outfield)
    - Compute Euclidean distance from each candidate to goal center
    - Rank by distance (ascending) → top1 = nearest, top3 = 3 nearest
    - Compare against true receiver label
- Aggregate top1/top3 across folds.
- No shot prediction for this baseline (receiver only).

Returns: Same format but only `receiver` metrics populated. `shot_*` fields set to None or omitted.

### 1.4 Create `corner_prediction/baselines/xgboost_baseline.py`

**Function: `xgboost_baseline_lomo(dataset, seed=42) -> Dict`**

Feature engineering (per graph → 1 feature vector):
```
# Team-based aggregates (attackers and defenders separately)
- mean_atk_x, mean_atk_y, std_atk_x, std_atk_y
- mean_def_x, mean_def_y, std_def_x, std_def_y
- mean_atk_vx, mean_atk_vy, mean_atk_speed
- mean_def_vx, mean_def_vy, mean_def_speed
- max_atk_speed, max_def_speed
- speed_differential (mean_atk_speed - mean_def_speed)
- n_attackers_in_box (x > 35, |y| < 20)
- n_defenders_in_box
- corner_side (graph-level feature)
- detection_rate
- mean_distance_atk_to_goal
- mean_distance_def_to_goal
```
Total: ~22-25 features.

Training per fold:
- Build feature matrix from train graphs
- Train XGBClassifier with scale_pos_weight = n_neg/n_pos
- Predict on test graphs
- Compute AUC, F1

Returns: Same format as lomo_cv() but only shot metrics (no receiver).

### 1.5 Create `corner_prediction/baselines/mlp_baseline.py`

**Function: `mlp_baseline_lomo(dataset, seed=42, device=None) -> Dict`**

Architecture:
```python
class ShotMLP(nn.Module):
    """Flatten 22 players × 13 features → 286 → MLP → P(shot)."""
    def __init__(self, input_dim=286, hidden_dim=64, dropout=0.3):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
```

Training per fold:
- Sort nodes by position (ensure consistent ordering — already ordered in graph construction)
- Flatten: [22, 13] → [286]
- Train with BCE loss + class weights, Adam, early stopping (patience=20)
- Evaluate: AUC, F1

Returns: Same format, shot metrics only.

## Step 2: Create run_baselines.py entry point

### 2.1 Create `corner_prediction/baselines/run_baselines.py`

CLI entry point:
```
python -m corner_prediction.baselines.run_baselines [OPTIONS]

Options:
  --baseline {random,heuristic,xgboost,mlp,all}  (default: all)
  --seed INT (default: 42)
  --no-gpu
  --output-dir PATH (default: results/corner_prediction)
```

For each baseline:
1. Load dataset
2. Run baseline-specific LOMO evaluation
3. Save results using `save_results()` from evaluate.py
4. Print comparison table

### 2.2 Comparison table printer

**Function: `print_baseline_comparison(baseline_results: Dict[str, Dict]) -> None`**

Prints a table comparing all baselines + main model results (if available):
```
====================================================================
Baseline Comparison Results
====================================================================
                          Receiver              Shot (AUC)
Model                 Top1      Top3      Oracle  Pred  Uncond
--------------------------------------------------------------------
Random               0.147     0.441       0.500   —     —
Heuristic            0.xxx     0.xxx         —     —     —
XGBoost (aggregate)    —         —         0.xxx   —     —
MLP (flat)             —         —         0.xxx   —     —
GNN unconditional      —         —           —     —    0.xxx
GNN two-stage        0.xxx     0.xxx       0.xxx 0.xxx 0.xxx
====================================================================
```

## Step 3: Create SLURM submission script

### 3.1 Create `scripts/slurm/run_baselines.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=short           # CPU partition for baselines 1-3
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --error=logs/baselines_%A_%a.err
```

Array tasks:
- 0 = Random baseline
- 1 = Heuristic receiver
- 2 = XGBoost aggregate
- 3 = MLP flat (needs GPU → override partition in submission)
- 4 = All baselines (sequential, for convenience)

Submit:
```bash
# All CPU baselines:
sbatch --array=0-2 scripts/slurm/run_baselines.sbatch

# MLP (needs GPU):
sbatch --array=3 --partition=dgpu --gres=gpu:1 scripts/slurm/run_baselines.sbatch

# All at once:
sbatch --array=4 scripts/slurm/run_baselines.sbatch
```

## Step 4: Write tests

### 4.1 Create `tests/corner_prediction/test_baselines.py`

Tests for each baseline using synthetic data (matching test_models.py patterns):

1. **test_random_baseline_returns_correct_format** — Run on 3-graph synthetic dataset, verify output has `per_fold`, `aggregated` keys
2. **test_random_receiver_top1_near_chance** — With synthetic data where n_candidates=10, top1 should be ~0.10
3. **test_heuristic_picks_nearest_to_goal** — Construct graph where one attacker is clearly nearest to goal, verify it's predicted
4. **test_xgboost_feature_extraction** — Verify feature vector has correct dimensionality
5. **test_xgboost_runs_lomo_fold** — Verify XGBoost trains and evaluates on one fold
6. **test_mlp_forward_pass** — ShotMLP produces [B, 1] output for batch of flattened graphs
7. **test_mlp_runs_lomo_fold** — Verify MLP trains and evaluates on one fold
8. **test_all_baselines_results_format** — Verify all baselines return compatible result dicts

## Step 5: Integration with run_all.py

### 5.1 Add `--baselines` flag to `corner_prediction/run_all.py`

Add to the mutually exclusive group:
```python
group.add_argument("--baselines", type=str, default=None,
                   choices=["random", "heuristic", "xgboost", "mlp", "all"],
                   help="Run baseline comparisons")
```

Wire it to call the baselines run_baselines logic.

### 5.2 Update SLURM array in `run_corner_prediction.sbatch`

Add array indices 8-12 for baselines (alternative to separate sbatch):
- Actually, keep separate sbatch file since resource requirements differ.

---

## File Checklist

| # | File | Action |
|---|------|--------|
| 1 | `corner_prediction/baselines/__init__.py` | Create |
| 2 | `corner_prediction/baselines/random_baseline.py` | Create |
| 3 | `corner_prediction/baselines/heuristic_receiver.py` | Create |
| 4 | `corner_prediction/baselines/xgboost_baseline.py` | Create |
| 5 | `corner_prediction/baselines/mlp_baseline.py` | Create |
| 6 | `corner_prediction/baselines/run_baselines.py` | Create |
| 7 | `scripts/slurm/run_baselines.sbatch` | Create |
| 8 | `tests/corner_prediction/test_baselines.py` | Create |
| 9 | `corner_prediction/run_all.py` | Edit — add --baselines flag |

## Test Criteria

- [ ] `pytest tests/corner_prediction/test_baselines.py` passes
- [ ] `python -m corner_prediction.baselines.run_baselines --baseline random --seed 42` runs to completion
- [ ] `python -m corner_prediction.baselines.run_baselines --baseline heuristic --seed 42` runs to completion
- [ ] `python -m corner_prediction.baselines.run_baselines --baseline xgboost --seed 42` runs to completion
- [ ] `python -m corner_prediction.baselines.run_baselines --baseline mlp --seed 42` runs to completion
- [ ] All result dicts have same top-level keys as `lomo_cv()` output
- [ ] SLURM script is syntactically valid
- [ ] Comparison table prints correctly
