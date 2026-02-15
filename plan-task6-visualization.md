# Task 6: Results Visualization — Implementation Plan

## Overview

Create thesis-ready visualizations for the two-stage corner kick prediction pipeline.
All figures saved as PDF for LaTeX inclusion. Output to `results/corner_prediction/figures/`.

## Files to Create

### 1. `corner_prediction/visualization/__init__.py`
Empty init file.

### 2. `corner_prediction/visualization/plot_corner.py` — Figure 1: Receiver Prediction Example

**Purpose:** Plot one corner kick on a pitch showing player positions, receiver probabilities, and predicted vs true receiver.

**Implementation:**
- Use `mplsoccer.Pitch(pitch_type='custom', pitch_length=105, pitch_width=68)` with center-origin coords mapped to mplsoccer's coordinate system.
- Load `extracted_corners.pkl` to get raw player positions (in meters, center-origin).
- Find a "good" example corner: one with `has_receiver_label=True` AND `lead_to_shot=True` AND high `detection_rate`.
- Load the trained model's per-fold predictions from LOMO results to get receiver probabilities for the chosen corner, OR run a single forward pass on the corresponding graph.
- **Approach: run a single forward pass** on the graph through the trained model. This requires loading the model and the graph. Simpler: use the `per_graph` data from the LOMO results which contains per-graph predictions.
- **Refined approach:** The LOMO `per_fold.receiver.per_graph` only stores `{top1, top3, n_candidates}` — no probabilities. So we need to either:
  (a) Run a forward pass to get probabilities, or
  (b) Plot without probability heatmap, just showing true receiver + predicted top-1.
- **Decision:** (a) is more informative. Build a helper that loads the pretrained model, runs a forward pass on one graph, and returns receiver probabilities. Only needs CPU.

**Plot elements:**
- Half-pitch view (attacking half only, where the action is).
- Attacking players: circles colored by receiver probability (colormap, e.g., Reds).
- Defending players: circles in a different color (blue/gray).
- Ball position: marked distinctly.
- True receiver: bold outline / star marker.
- Predicted receiver (argmax of probs): different marker.
- Arrow from corner taker to predicted receiver.
- Colorbar showing probability scale.
- Legend identifying attacking/defending/ball/true/predicted.
- Corner side annotation.

**Function signature:**
```python
def plot_receiver_example(
    corner_record: dict,
    receiver_probs: np.ndarray,  # [22] probabilities
    receiver_mask: np.ndarray,   # [22] bool
    output_path: str = None,
    show: bool = False,
) -> matplotlib.figure.Figure
```

### 3. `corner_prediction/visualization/plot_ablation.py` — Figure 2: Ablation Comparison

**Purpose:** Grouped bar chart showing Top-1 accuracy, Top-3 accuracy across all 5 ablation configs.

**Implementation:**
- Load `ablation_all_*.pkl` (latest by timestamp).
- Extract per-ablation aggregated receiver metrics (`top1_mean/std`, `top3_mean/std`).
- Grouped bar chart: x-axis = ablation config, bars = Top-1 and Top-3.
- Error bars from std.
- Horizontal dashed lines at random baselines (top1 ~0.147, top3 ~0.441).
- Config names on x-axis as readable labels.
- Clean LaTeX-compatible formatting.

**Function signature:**
```python
def plot_ablation_comparison(
    ablation_results: dict,  # from ablation_all pickle
    output_path: str = None,
    show: bool = False,
) -> matplotlib.figure.Figure
```

### 4. `corner_prediction/visualization/plot_shot_auc.py` — Figure 3: Shot Prediction AUC Comparison

**Purpose:** Bar chart comparing shot AUC across conditions: unconditional, oracle receiver, predicted receiver, plus ablation position_only, plus XGBoost baseline (if available).

**Implementation:**
- Load `lomo_pretrained_*.pkl` for oracle/predicted/unconditional.
- Load `ablation_all_*.pkl` for position_only shot AUC.
- Optionally load `baseline_xgboost_*.pkl` if it exists.
- Bar chart: x-axis = condition, y-axis = AUC.
- Error bars from std across LOMO folds.
- Horizontal line at AUC=0.50 (random).
- Color-code: GNN conditions in one color family, baselines in another.

**Function signature:**
```python
def plot_shot_auc_comparison(
    lomo_results: dict,
    ablation_results: dict = None,
    baseline_results: dict = None,  # optional
    output_path: str = None,
    show: bool = False,
) -> matplotlib.figure.Figure
```

### 5. `corner_prediction/visualization/plot_two_stage.py` — Figure 4: Two-Stage Benefit

**Purpose:** Scatter plot: x = receiver prediction accuracy per fold, y = conditional shot AUC per fold.

**Implementation:**
- Load `lomo_pretrained_*.pkl`.
- From `per_fold`: extract `receiver.top3_acc` and `shot_predicted.auc` for each fold.
- Scatter plot: one point per fold (10 points).
- Add trend line (linear regression) if meaningful.
- Annotate with fold number or held-out match.
- Compare: also plot unconditional AUC as horizontal line.

**Function signature:**
```python
def plot_two_stage_benefit(
    lomo_results: dict,
    output_path: str = None,
    show: bool = False,
) -> matplotlib.figure.Figure
```

### 6. `corner_prediction/visualization/plot_sensitivity.py` — Figure 5: Detection Rate Sensitivity

**Purpose:** Line plot showing model performance vs detection rate threshold.

**Implementation:**
- Load `lomo_pretrained_*.pkl` — need per-graph predictions with detection_rate.
- **Problem:** The per_fold results don't store per-graph detection rates alongside predictions.
- **Solution:** Load `extracted_corners.pkl` for detection rates, match by corner_id. Then from `per_fold[fold_idx].shot_oracle.probs/labels`, pair each prediction with its corner's detection rate.
- **Issue:** per_fold.shot_oracle.probs is a flat list — order matches the test_data order for that fold. We need to reconstruct which corners are in each fold.
- **Approach:** Reconstruct LOMO splits from dataset (same logic as evaluate.py), match detection_rate from the graph metadata (`g.detection_rate`), then pair with per-fold predictions.
- Plot: x = detection rate threshold (e.g., 0.3, 0.4, ..., 0.9), y = AUC computed only on corners above threshold. Show sample size at each threshold.

**Function signature:**
```python
def plot_detection_sensitivity(
    lomo_results: dict,
    dataset,  # PyG dataset with detection_rate metadata
    output_path: str = None,
    show: bool = False,
) -> matplotlib.figure.Figure
```

### 7. `corner_prediction/visualization/plot_permutation.py` — Bonus: Permutation Test Null Distribution

**Purpose:** Histogram of null distribution with real metric marked. Not in original Task 6 spec but highly useful for thesis.

**Implementation:**
- Load `perm_receiver_*.pkl` and `perm_shot_*.pkl`.
- Two subplots: receiver (top3_acc) and shot (AUC).
- Histogram of null_distribution, vertical line at real_metric.
- Annotate p-value.

### 8. `corner_prediction/visualization/generate_all.py` — Orchestrator

**Purpose:** Load all results, generate all figures, save as PDFs.

**Implementation:**
- CLI script: `python -m corner_prediction.visualization.generate_all`
- Args: `--output-dir`, `--show`, `--results-dir`
- Finds latest result files by timestamp.
- Calls each plot function.
- Saves to `results/corner_prediction/figures/`.
- Prints summary of generated files.

**Function signature:**
```python
def find_latest_result(results_dir: Path, prefix: str) -> Path
def generate_all(results_dir: Path, output_dir: Path, show: bool = False) -> None
```

### 9. `scripts/slurm/run_visualizations.sbatch` — SLURM Script

**Implementation:**
- CPU-only job (no GPU needed).
- Partition: `short` (visualization is fast).
- Time: 00:30:00 (generous for PDF generation).
- Memory: 4G (loading pickles + matplotlib).
- Runs: `python -u -m corner_prediction.visualization.generate_all`

```bash
#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --partition=short
#SBATCH --account=researchers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err
```

### 10. Integration: Add `--visualize` flag to `corner_prediction/run_all.py`

Add a new mutually exclusive group option:
```python
group.add_argument("--visualize", action="store_true", help="Generate all figures")
```

And in the dispatch logic:
```python
elif args.visualize:
    from corner_prediction.visualization.generate_all import generate_all
    generate_all(Path(args.output_dir), Path(args.output_dir) / "figures", show=False)
```

## Execution Order

1. Create `corner_prediction/visualization/__init__.py`
2. Create `corner_prediction/visualization/plot_corner.py` (Figure 1)
3. Create `corner_prediction/visualization/plot_ablation.py` (Figure 2)
4. Create `corner_prediction/visualization/plot_shot_auc.py` (Figure 3)
5. Create `corner_prediction/visualization/plot_two_stage.py` (Figure 4)
6. Create `corner_prediction/visualization/plot_sensitivity.py` (Figure 5)
7. Create `corner_prediction/visualization/plot_permutation.py` (Bonus Figure)
8. Create `corner_prediction/visualization/generate_all.py` (Orchestrator)
9. Create `scripts/slurm/run_visualizations.sbatch` (SLURM)
10. Edit `corner_prediction/run_all.py` to add `--visualize` flag
11. Test: Run `generate_all.py` locally to verify figures render

## Design Decisions

- **matplotlib backend:** Use `Agg` (non-interactive) for SLURM compatibility.
- **Figure size:** 8×5 inches for bar charts, 10×7 for pitch plot. Standard for thesis.
- **Font size:** 12pt default, matching typical LaTeX body text.
- **Color palette:** Use colorblind-friendly colors (seaborn's "colorblind" palette).
- **PDF output:** `fig.savefig(path, bbox_inches='tight', dpi=300)` for vector quality.
- **Graceful degradation:** If a result file is missing (e.g., baselines), skip that figure and warn.
- **No mplsoccer for pitch:** Actually, the coordinate system is center-origin (-52.5 to 52.5, -34 to 34). mplsoccer expects (0, 120) or similar. Simpler to draw the pitch manually with matplotlib rectangles/arcs, or use mplsoccer with `pitch_type='custom'` and map coordinates. Decision: use mplsoccer's `VerticalPitch(pitch_type='skillcorner', half=True)` — mplsoccer natively supports skillcorner coordinates.

## Test Criteria

- All 5 (+ bonus) figures generate without errors.
- PDFs are valid and readable.
- Bar charts have correct error bars matching the result data.
- Pitch plot shows correct player positions from extracted_corners.pkl.
- SLURM script submits and completes successfully.
- `--visualize` flag works in run_all.py.
