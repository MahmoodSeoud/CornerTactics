# Multi-Class Outcome Baseline Documentation

**Author**: mseo
**Date**: November 4, 2025
**Implementation**: Day 6.5 - TacticAI Implementation Plan
**Branch**: `feature/tacticai-multiclass-outcome-baselines`

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Definition](#problem-definition)
3. [Dataset](#dataset)
4. [Baseline Models](#baseline-models)
5. [Training Pipeline](#training-pipeline)
6. [Results](#results)
7. [Implementation Details](#implementation-details)
8. [Quick Start](#quick-start)

---

## Overview

This document describes the implementation of **multi-class outcome prediction baselines** for corner kick scenarios. The goal is to predict what happens after a corner kick using only the static player positions at the moment of the kick.

**Three baseline models** are implemented:
1. **RandomOutcomeBaseline**: Uniform random prediction (sanity check)
2. **XGBoostOutcomeBaseline**: Tree-based model with engineered graph-level features
3. **MLPOutcomeBaseline**: Neural network with flattened player features

**Task**: 3-class classification
- **Shot** (Goal + Shot merged): 18.2%
- **Clearance**: 52.0%
- **Possession** (Possession + Loss merged): 29.9%

**Key Finding**: Static player positions alone achieve **50.1% accuracy** (XGBoost) and **44.7% accuracy** (MLP), significantly above the 33% random baseline, but below TacticAI targets (55-65%). This suggests that **temporal dynamics** (velocity, momentum) are necessary for accurate outcome prediction.

---

## Problem Definition

### Task: Multi-Class Outcome Prediction

Given static player positions at corner kick moment, predict the corner outcome:

```
Input:  Player positions [22 nodes × 14 features]
          ↓
       Model
          ↓
Output: P(Shot), P(Clearance), P(Possession)
```

### Class Definitions (3-Class)

| Class ID | Class Name | Description | Percentage |
|----------|------------|-------------|------------|
| 0 | **Shot** | Goal scored OR shot attempt within 20s | 18.2% (1,056) |
| 1 | **Clearance** | Defensive clearance or interception | 52.0% (3,021) |
| 2 | **Possession** | Attacking team retains possession or loses ball | 29.9% (1,737) |

**Total**: 5,814 corner kick scenarios

### Why 3-Class Instead of 4-Class?

**Original 4-class problem** had severe imbalance:
- Goal: 1.3% (76 samples) ← **Too rare!**
- Shot: 16.9% (980 samples)
- Clearance: 52.0% (3,021 samples)
- Possession: 29.9% (1,737 samples)

**Problem**: All three models achieved **Goal F1 = 0.000** because the Goal class was too rare to predict from static positions alone.

**Solution**: Merge Goal + Shot into single "Shot" class (18.2%), treating both as "dangerous situations."

**Impact**: Macro F1 improved ~30% for both XGBoost and MLP after merging.

---

## Dataset

### Data Source

**File**: `data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl`

**Origin**: StatsBomb 360 freeze frame data with temporal augmentation
**Coverage**: 100% receiver labels (5,814/5,814 graphs)

### Dataset Statistics

```
Total graphs:               5,814
Total corners (unique):     1,118 (before augmentation)
Augmentation factor:        5.2×

Train/Val/Test split:
  Train: 4,066 (69.9%)
  Val:     871 (15.0%)
  Test:    877 (15.1%)

Average nodes per graph:    19.0 players
Average edges per graph:    166.0 connections
Node features:              14 dimensions
Edge features:              6 dimensions
```

### Class Distribution

```
Outcome Class Distribution (3-Class):
  0. Shot        : 1,056 samples (18.2%)
  1. Clearance   : 3,021 samples (52.0%)
  2. Possession  : 1,737 samples (29.9%)

Split stratification (Shot rate):
  Train: 18.1%
  Val:   18.3%
  Test:  18.6%
```

### Node Features (14 Dimensions)

Each player (node) has 14-dimensional feature vector:

| Category | Features | Dimension |
|----------|----------|-----------|
| **Spatial** | x, y, distance_to_goal, distance_to_ball_target | 4 |
| **Kinematic** | vx, vy, velocity_magnitude, velocity_angle | 4 |
| **Contextual** | angle_to_goal, angle_to_ball, team_flag, in_penalty_box | 4 |
| **Density** | num_players_within_5m, local_density_score | 2 |

**Important**: Velocity features (vx, vy) are **masked to zero** because StatsBomb 360 freeze frames are single-frame snapshots without temporal tracking.

### Temporal Augmentation

**Method**: US Soccer Federation approach (5 temporal frames + mirror augmentation)

```
Original corner at t=0
    ↓
Generate 5 temporal frames:
  t = -2s, -1s, 0s, +1s, +2s
    ↓
Apply position perturbations:
  - Jitter: ±0.5m Gaussian noise
  - Mirror: Flip across vertical axis
    ↓
5,814 augmented graphs (5.2× increase)
```

**Rationale**: Simulate temporal context from static snapshots to increase dataset size.

**Limitation**: Simulated velocities cannot fully replace real tracking data.

---

## Baseline Models

### 1. RandomOutcomeBaseline

**Purpose**: Sanity check (lower bound performance)

**Architecture**:
```python
def forward(self, x, batch):
    batch_size = batch.max().item() + 1
    logits = torch.randn(batch_size, 3)  # Uniform random
    return logits
```

**Expected Performance**:
- Accuracy: 33% (1/3 uniform random)
- Macro F1: ~0.33

**Implementation**: `src/models/baselines.py:RandomOutcomeBaseline`

---

### 2. XGBoostOutcomeBaseline

**Purpose**: Tree-based model with hand-crafted graph-level features

**Architecture**:
```
Input [22 nodes × 14 features]
         ↓
   Graph-Level Feature Extraction (29 dimensions)
         ↓
   XGBoost (500 trees, max_depth=6)
         ↓
   P(Shot), P(Clearance), P(Possession)
```

**Graph-Level Features (29 Dimensions)**:

| Category | Features | Count |
|----------|----------|-------|
| **Position Statistics** | Mean x/y, std x/y (attacking/defending) | 8 |
| **Formation** | Team compactness, defensive line height | 2 |
| **Spatial** | Avg distance to goal, avg distance to ball target | 2 |
| **Density** | Players in box (attack/defense), goal area crowding | 3 |
| **Angle** | Mean angle to goal, mean angle to ball | 2 |
| **Team Balance** | Attacking/defending player count ratio | 1 |
| **Zone Coverage** | Players in penalty box, 6-yard box, etc. | 11 |

**Hyperparameters**:
```python
XGBoostClassifier(
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
```

**Expected Performance**:
- Accuracy: 55-65%
- Macro F1: > 0.50

**Implementation**: `src/models/baselines.py:XGBoostOutcomeBaseline`

---

### 3. MLPOutcomeBaseline

**Purpose**: Neural network baseline with learned features

**Architecture**:
```
Input [22 nodes × 14 features]
         ↓
   Flatten + Zero-Padding (→ 308 dims)
         ↓
   Linear(308 → 512) + ReLU + Dropout(0.25)
         ↓
   Linear(512 → 256) + ReLU + Dropout(0.25)
         ↓
   Linear(256 → 128) + ReLU + Dropout(0.25)
         ↓
   Linear(128 → 3)
         ↓
   P(Shot), P(Clearance), P(Possession)
```

**Model Parameters**: 322,819

**Why Flattening?**
- Converts graph (variable nodes) → fixed vector (308 dims)
- Zero-pads graphs with < 22 players
- Loses spatial structure (unlike GNN)

**Training Details**:
- **Optimizer**: Adam (lr=5e-4, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss with **SQRT class weights**
- **LR Scheduler**: Cosine annealing (T_max=15000 steps)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: Patience=5000 steps (no improvement in Val Macro F1)
- **Batch Size**: 128
- **Training Steps**: 15,000 (stopped early at ~6,000)

**Class Weighting Strategy**:

Raw balanced class weights caused training collapse:
```python
# Problem: Extreme weights
class_weights_raw = [1.85, 0.63, 1.14]  # Goal weight was 33.88 in 4-class!
```

**Solution**: Use SQRT of balanced weights to reduce extremes:
```python
class_weights = np.sqrt(class_weights_raw)
# Result: [1.36, 0.79, 1.07] ← Much more reasonable
```

**Expected Performance**:
- Accuracy: 60-70%
- Macro F1: > 0.55

**Implementation**: `src/models/baselines.py:MLPOutcomeBaseline`

---

## Training Pipeline

### Script: `scripts/training/train_outcome_baselines.py`

**Main Flow**:
```python
def main(args):
    # 1. Load dataset with outcome labels
    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        mask_velocities=True  # Mask vx, vy to zero
    )

    # 2. Train Random Baseline (no training needed)
    if args.models in ['all', 'random']:
        results['random'] = train_random_baseline(test_loader, args.device)

    # 3. Train XGBoost Baseline
    if args.models in ['all', 'xgboost']:
        results['xgboost'] = train_xgboost_baseline(
            train_loader, val_loader, test_loader, args.device
        )

    # 4. Train MLP Baseline
    if args.models in ['all', 'mlp']:
        test_metrics, history, model = train_mlp_baseline(
            train_loader, val_loader, test_loader, args.device, args.num_steps
        )
        results['mlp'] = test_metrics

        # Save MLP model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'test_metrics': test_metrics
        }, 'results/baselines/outcome_mlp_model.pt')

    # 5. Save results to JSON
    save_results(results, Path(args.output_dir))
```

### CLI Arguments

```bash
python scripts/training/train_outcome_baselines.py \
    --graph-path data/graphs/.../statsbomb_temporal_augmented_with_receiver.pkl \
    --batch-size 128 \
    --num-steps 15000 \
    --device cuda \
    --models all \
    --output-dir results/baselines \
    --random-seed 42
```

### SLURM Execution

**Scripts**: `scripts/slurm/train_outcome_baselines_{a100,v100,h100}.sh`

```bash
#!/bin/bash
#SBATCH --job-name=outcome_a100
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/outcome_baselines_a100_%j.out
#SBATCH --error=logs/outcome_baselines_a100_%j.err

conda activate robo
cd /home/mseo/CornerTactics

python scripts/training/train_outcome_baselines.py \
    --models all \
    --batch-size 128 \
    --num-steps 15000 \
    --device cuda \
    --output-dir results/baselines \
    --random-seed 42
```

**Submit Job**:
```bash
sbatch scripts/slurm/train_outcome_baselines_a100.sh
```

**Monitor**:
```bash
squeue -u $USER
tail -f logs/outcome_baselines_a100_31089.out
```

---

## Results

### Training History (Job 31089 - A100)

**Dataset**: 5,814 graphs (3-class: Shot/Clearance/Possession)

**Training Time**:
- Random: < 1 second (no training)
- XGBoost: ~30 seconds (500 trees)
- MLP: ~3 minutes (6,000 steps with early stopping)

**GPU**: NVIDIA A100-PCIE-40GB

---

### Final Test Set Performance (3-Class)

| Model | Accuracy | Macro F1 | Shot F1 | Clearance F1 | Possession F1 |
|-------|----------|----------|---------|--------------|---------------|
| **Random** | 33.2% | 0.319 | 0.214 | 0.391 | 0.353 |
| **XGBoost** | **50.1%** | **0.419** | 0.207 | **0.604** | 0.445 |
| **MLP** | 44.7% | 0.385 | **0.219** | 0.566 | 0.369 |

**Key Observations**:

1. **XGBoost outperforms MLP** (50.1% vs 44.7% accuracy)
   - Hand-crafted features better capture static patterns
   - Tree-based models excel with engineered features

2. **Shot class remains difficult** (F1 ~0.21 for both models)
   - Even after merging Goal+Shot, still hard to predict
   - Static positions lack momentum/dynamics needed for danger prediction

3. **Clearance class easiest** (F1 ~0.60 for XGBoost)
   - Defensive formations more predictable from static positions
   - Higher baseline class frequency (52%) helps

4. **Below TacticAI targets** (expected Macro F1 > 0.50)
   - Gap: ~15-20% below target
   - Root cause: Missing velocity/temporal context

---

### Confusion Matrices

#### XGBoost Confusion Matrix (Test Set)
```
Predicted:       Shot  Clearance  Possession
Actual:
  Shot            25      104         34       (163 samples)
  Clearance       35      294         80       (409 samples)
  Possession      18      167        120       (305 samples)
```

**Insights**:
- **Shot → Clearance confusion** (104/163 = 63.8%)
  - Model over-predicts Clearance (majority class bias)
- **Clearance accuracy** (294/409 = 71.9%)
  - Best performing class

#### MLP Confusion Matrix (Test Set)
```
Predicted:       Shot  Clearance  Possession
Actual:
  Shot            34       99         30       (163 samples)
  Clearance       61      263         85       (409 samples)
  Possession      52      158         95       (305 samples)
```

**Insights**:
- Similar confusion pattern as XGBoost
- Slightly better Shot recall (34 vs 25)
- Worse Clearance precision due to overprediction

---

### MLP Training Curve

**Best Validation Macro F1**: 0.451 (at step 1000)

```
Step   Val Acc   Macro F1   Shot F1   Clearance F1
----   -------   --------   -------   ------------
  500   49.5%     0.421      0.231       0.645
 1000   53.2%     0.451      0.295       0.671  ← Best
 1500   47.9%     0.391      0.165       0.620
 2000   48.9%     0.389      0.142       0.640
 ...
 6000   48.9%     0.370      0.081       0.644  ← Early stop
```

**Early Stopping Triggered**: Step 6000/15000 (no improvement for 5000 steps)

**Observations**:
- Peaked early (step 1000)
- Gradual overfitting after peak
- Healthy convergence pattern

---

### Performance Comparison: 4-Class vs 3-Class

#### Why Merging Goal+Shot Improved Performance

| Metric | 4-Class (Job 31085) | 3-Class (Job 31089) | Improvement |
|--------|---------------------|---------------------|-------------|
| **XGBoost Macro F1** | 0.325 | 0.419 | +28.9% |
| **XGBoost Accuracy** | 50.4% | 50.1% | -0.6% |
| **MLP Macro F1** | 0.294 | 0.385 | +31.0% |
| **MLP Accuracy** | 43.3% | 44.7% | +3.2% |

**Key Insight**: Macro F1 improved ~30% by merging Goal into Shot, even though accuracy stayed similar. This is because Goal class (1.3%) was unpredictable (F1=0.0), dragging down the macro average.

---

## Implementation Details

### Critical Training Fixes

#### Problem 1: MLP Training Collapse

**First training attempt** (Job 31084):
```
Test Accuracy: 13.5% (worse than 24.6% random!)
Predictions: 643 Goals, 25 Clearances ← Extreme bias
```

**Root Cause**: Raw balanced class weights too extreme
```python
class_weights_raw = compute_class_weight('balanced', classes, y)
# Goal weight was 33.88! → Model over-predicted Goals
```

**Fix**: Use SQRT of class weights
```python
class_weights = np.sqrt(class_weights_raw)
# Goal: 33.88 → 5.82 (much more reasonable)
```

**Additional Stabilization**:
```python
# 1. Cosine annealing LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_steps, eta_min=lr/10
)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Early stopping
if no_improvement_steps >= patience:
    break
```

**Result**: Accuracy improved from 13.5% → 44.7%

---

#### Problem 2: KeyError in History Tracking

**After refactoring to 3-class**, training crashed:
```python
KeyError: 'goal_f1'
File "src/models/baselines.py", line 1564, in train_mlp_outcome
    history['val_goal_f1'].append(val_metrics['goal_f1'])
```

**Fix**: Update history keys to match 3-class names
```python
history = {
    'train_loss': [],
    'val_accuracy': [],
    'val_macro_f1': [],
    'val_shot_f1': [],       # Changed from val_goal_f1
    'val_clearance_f1': [],  # Changed from val_shot_f1
    'steps': []
}
```

---

### Evaluation Metrics

**Function**: `src/models/baselines.py:evaluate_outcome_baseline()`

```python
def evaluate_outcome_baseline(model, test_loader, device):
    """
    Evaluate outcome baseline model on test set.

    Returns:
        metrics (dict):
            - accuracy: Overall accuracy
            - macro_f1: Macro-averaged F1 (unweighted)
            - weighted_f1: Weighted F1 (by class support)
            - macro_precision, macro_recall
            - shot_f1, clearance_f1, possession_f1
            - confusion_matrix: [3 × 3] matrix
    """
    all_preds = []
    all_labels = []

    for batch in test_loader:
        logits = model(batch.x, batch.batch)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.outcome_class_label.cpu().numpy())

    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix
    )

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'macro_precision': precision_score(all_labels, all_preds, average='macro'),
        'macro_recall': recall_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }

    # Per-class F1 scores
    class_f1 = f1_score(all_labels, all_preds, average=None)
    metrics['shot_f1'] = class_f1[0]
    metrics['clearance_f1'] = class_f1[1]
    metrics['possession_f1'] = class_f1[2]

    return metrics
```

---

### Data Loading

**Function**: `src/data/receiver_data_loader.py:load_receiver_dataset()`

```python
def load_receiver_dataset(
    graph_path: str,
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    mask_velocities: bool = True
) -> Tuple[ReceiverCornerDataset, DataLoader, DataLoader, DataLoader]:
    """
    Load dataset with outcome labels and create train/val/test loaders.

    Args:
        graph_path: Path to pickle file with receiver and outcome labels
        batch_size: Batch size for data loaders
        test_size: Test set fraction (0.15 = 15%)
        val_size: Validation set fraction (0.15 = 15%)
        random_state: Random seed for reproducible splits
        mask_velocities: If True, set vx, vy to zero (no velocity data)

    Returns:
        dataset, train_loader, val_loader, test_loader
    """
    dataset = ReceiverCornerDataset(graph_path, mask_velocities=mask_velocities)
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    return dataset, train_loader, val_loader, test_loader
```

**Outcome Class Mapping** (3-Class):
```python
# src/data/receiver_data_loader.py
OUTCOME_CLASS_MAPPING = {
    "Goal": 0,          # Merged with Shot → Shot (~1.3%)
    "Shot": 0,          # Shot (~16.9%) → Combined ~18.2%
    "Clearance": 1,     # ~52.0% (common)
    "Possession": 2,    # ~10.5% + Loss ~19.4% = ~29.9% (merged)
    "Loss": 2           # Merged into Possession
}

OUTCOME_CLASS_NAMES = ["Shot", "Clearance", "Possession"]
```

---

## Quick Start

### 1. Train All Baselines (GPU)

```bash
# Submit to A100 GPU
sbatch scripts/slurm/train_outcome_baselines_a100.sh

# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/outcome_baselines_a100_31089.out
```

### 2. Train Individual Models

```bash
# Train only XGBoost
python scripts/training/train_outcome_baselines.py \
    --models xgboost \
    --batch-size 128 \
    --device cuda

# Train only MLP (5000 steps)
python scripts/training/train_outcome_baselines.py \
    --models mlp \
    --batch-size 128 \
    --num-steps 5000 \
    --device cuda
```

### 3. Check Results

```bash
# View results
cat results/baselines/outcome_xgboost_results.json
cat results/baselines/outcome_mlp_results.json

# Load saved MLP model
python -c "
import torch
checkpoint = torch.load('results/baselines/outcome_mlp_model.pt')
print('Best Val Macro F1:', max(checkpoint['history']['val_macro_f1']))
"
```

### 4. Test on Custom Data

```python
import torch
from src.models.baselines import MLPOutcomeBaseline
from src.data.receiver_data_loader import load_receiver_dataset

# Load model
model = MLPOutcomeBaseline(num_features=14, num_players=22, num_classes=3)
checkpoint = torch.load('results/baselines/outcome_mlp_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
_, _, _, test_loader = load_receiver_dataset(
    graph_path='data/graphs/.../statsbomb_temporal_augmented_with_receiver.pkl',
    batch_size=32
)

# Predict
batch = next(iter(test_loader))
logits = model(batch.x, batch.batch)
probs = torch.softmax(logits, dim=1)
print(f"Predictions: {probs}")
```

---

## Summary

### Key Findings

1. **XGBoost outperforms MLP** with static features (50.1% vs 44.7% accuracy)
2. **Shot class remains difficult** (F1 ~0.21) even after merging Goal+Shot
3. **Below TacticAI targets** by ~15-20% due to missing temporal context
4. **Merging Goal+Shot improved Macro F1 by ~30%** (class imbalance fix)
5. **Static positions alone are insufficient** for accurate outcome prediction

### Next Steps

1. **Implement GNN baselines** to leverage graph structure
2. **Add temporal features** (velocity, acceleration from tracking data)
3. **Try sequence models** (LSTM, Transformer) for temporal dynamics
4. **Investigate Shot class errors** (why so hard to predict?)
5. **Compare with TacticAI paper results** (need velocity data)

---

## File Structure

```
CornerTactics/
├── src/
│   ├── data/
│   │   └── receiver_data_loader.py       # Load graphs with outcome labels
│   └── models/
│       └── baselines.py                  # Random, XGBoost, MLP models
│
├── scripts/
│   ├── training/
│   │   └── train_outcome_baselines.py    # Training script
│   └── slurm/
│       ├── train_outcome_baselines_a100.sh
│       ├── train_outcome_baselines_v100.sh
│       └── train_outcome_baselines_h100.sh
│
├── results/
│   └── baselines/
│       ├── outcome_random_results.json
│       ├── outcome_xgboost_results.json
│       ├── outcome_mlp_results.json
│       └── outcome_mlp_model.pt          # Saved MLP checkpoint
│
├── logs/
│   └── outcome_baselines_a100_31089.out  # Training logs
│
└── docs/
    └── OUTCOME_BASELINE_DOCUMENTATION.md  # This file
```

---

## References

1. **TacticAI Implementation Plan**: `docs/TACTICAI_IMPLEMENTATION_PLAN.md` (Day 6.5)
2. **Dataset Documentation**: `docs/DATASET_DOCUMENTATION.md`
3. **Receiver Baseline Architecture**: `docs/BASELINE_ARCHITECTURE.md`
4. **Training Logs**: `logs/outcome_baselines_a100_31089.out`

---

**End of Document**
