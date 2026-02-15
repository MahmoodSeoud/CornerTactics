# Plan: Multi-Source Corner Kick Prediction Experiments

## Overview

Train and evaluate GNN models on the expanded multi-source dataset (3,078 corners)
to answer whether more data solves the corner prediction problem.

**Data**: Pre-built splits in `transfer_learning/data/multi_source_corners_dense_{train,val,test}.pkl`
**Model**: `TransferGNN` (CGConv backbone, 12 node features, 6 edge features, 128 hidden)
**Output**: `results/multi_source_experiments/`

---

## Files to create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `transfer_learning/multi_source_utils.py` | Shared infrastructure | ~250 |
| `transfer_learning/exp1_full_dataset.py` | Experiment 1: full dataset | ~200 |
| `transfer_learning/exp2_source_stratified.py` | Experiment 2: per-source eval | ~150 |
| `transfer_learning/exp3_source_ablation.py` | Experiment 3: train-source ablation | ~180 |
| `transfer_learning/exp4_feature_ablation.py` | Experiment 4: velocity ablation | ~200 |

No existing files are modified.

---

## Step 1: Create `transfer_learning/multi_source_utils.py`

Shared utilities extracted/adapted from phase3. All experiments import from here.

### 1.1 Imports and constants

```python
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "multi_source_experiments"

FEATURE_INDICES = {
    'x': 0, 'y': 1, 'vx': 2, 'vy': 3,
    'velocity_mag': 4, 'velocity_angle': 5,
    'dist_goal': 6, 'angle_goal': 7,
    'dist_ball': 8, 'angle_ball': 9,
    'attacking_team_flag': 10, 'potential_receiver': 11,
}
VELOCITY_FEATURES = [2, 3, 4, 5]
POSITION_FEATURES = [0, 1, 6, 7, 8, 9]
```

### 1.2 `TransferGNN` class

Copy from phase3 with one change: `head_hidden` default raised from 32 → 64.
Rationale: 3,078 samples (54× more) can support a larger head.

Signature unchanged:
```python
class TransferGNN(nn.Module):
    def __init__(self, node_features=12, edge_features=6, hidden_channels=128,
                 num_conv_layers=3, head_hidden=64, head_dropout=0.3,
                 freeze_backbone=True):
```

### 1.3 `load_splits()` function

```python
def load_splits() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load pre-built train/val/test splits."""
    train = pickle.load(open(DATA_DIR / "multi_source_corners_dense_train.pkl", "rb"))
    val = pickle.load(open(DATA_DIR / "multi_source_corners_dense_val.pkl", "rb"))
    test = pickle.load(open(DATA_DIR / "multi_source_corners_dense_test.pkl", "rb"))
    return train, val, test
```

### 1.4 `prepare_pyg_data()` function

Copied from phase3 `prepare_pyg_data()` (lines 165-202) with one addition:
store `source` metadata on each Data object for source-stratified eval.

```python
def prepare_pyg_data(corners: List[Dict]) -> List[Data]:
    data_list = []
    for sample in corners:
        graph = sample['graphs'][0]
        label = float(sample['labels']['shot_binary'])
        pyg_data = Data(
            x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
            y=torch.tensor([label], dtype=torch.float32)
        )
        pyg_data.match_id = sample['match_id']
        pyg_data.corner_time = sample.get('corner_time', 0.0)
        pyg_data.source = sample.get('source', 'unknown')
        data_list.append(pyg_data)
    return data_list
```

### 1.5 `compute_class_weights()`, `train_epoch()`, `evaluate()`

Copied verbatim from phase3 (lines 285-402). No changes needed — they are
data-size-agnostic.

### 1.6 `create_model()` factory function

```python
def create_model(pretrained: bool = False, freeze_backbone: bool = False,
                 backbone_path: str = "ussf_backbone_dense.pt",
                 head_hidden: int = 64, device: str = "cpu") -> TransferGNN:
    model = TransferGNN(
        head_hidden=head_hidden,
        freeze_backbone=freeze_backbone,
    ).to(device)
    if pretrained:
        state = torch.load(WEIGHTS_DIR / backbone_path, map_location=device)
        model.load_pretrained_backbone(state)
    return model
```

### 1.7 `run_training()` function

Adapted from phase3 `run_experiment()` (lines 405-552). Changes:
- `batch_size` default: 8 → 32
- `epochs` default: 50 → 100
- `patience` default: 10 → 15
- `head_hidden` parameter passed through
- Returns model along with result dict (for downstream eval)

```python
def run_training(
    train_data, val_data, test_data,
    pretrained=False, freeze_backbone=False,
    backbone_path="ussf_backbone_dense.pt",
    lr=1e-3, epochs=100, batch_size=32, patience=15,
    head_hidden=64, device=None, seed=42,
    label="experiment",
) -> Tuple[TransferGNN, Dict]:
```

### 1.8 `majority_baseline()` function

Copied from phase3 `run_majority_baseline()` (lines 555-617). No changes.

### 1.9 `bootstrap_auc_ci()` function

Copied from phase3 `bootstrap_ci()` (lines 620-663). No changes.

### 1.10 `permutation_test()` function (NEW)

Label-permutation test: shuffle labels N times, retrain head only each time,
build null AUC distribution.

```python
def permutation_test(
    model: TransferGNN, train_data: List[Data], test_data: List[Data],
    n_permutations: int = 20, epochs: int = 30, batch_size: int = 32,
    device: str = "cpu", seed: int = 42,
) -> Dict:
    """Label permutation test.

    For each permutation:
    1. Shuffle train labels
    2. Re-initialize and train head only (backbone frozen)
    3. Evaluate on (unshuffled) test set
    4. Record AUC

    Returns null_aucs list and p-value.
    """
```

Uses frozen backbone from the trained model to keep each permutation fast
(~30 epochs, head-only training). p-value = (n_null >= observed + 1) / (N + 1).

### 1.11 `save_results()` function

```python
def save_results(results: Dict, filename: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    pickle.dump(results, open(path, "wb"))
    print(f"Saved: {path}")
```

### 1.12 `print_summary_table()` function

Pretty-print results as a markdown-style table.

---

## Step 2: Create `transfer_learning/exp1_full_dataset.py`

**Question**: "Does 50× more data solve the corner prediction problem?"

### Conditions

| ID | Description | Pretrained | Frozen | LR | head_hidden |
|----|-------------|------------|--------|----|-------------|
| MS-A | USSF pretrained + frozen (linear probe) | Yes | Yes | 1e-3 | 64 |
| MS-B | USSF pretrained + fine-tuned | Yes | No | 1e-4 | 64 |
| MS-C | Random init (train from scratch) | No | No | 1e-3 | 64 |
| MS-D | Majority baseline | - | - | - | - |

Note: only dense adjacency (dense > normal was settled in Phase 3).

### Algorithm

```
for seed in [42, 123, 456, 789, 1234]:
    set_seed(seed)
    train, val, test = load_splits()
    train_data = prepare_pyg_data(train)
    val_data = prepare_pyg_data(val)
    test_data = prepare_pyg_data(test)

    for condition in [MS-A, MS-B, MS-C, MS-D]:
        model, result = run_training(...)

        # Bootstrap CI on test predictions
        ci = bootstrap_auc_ci(result['test_metrics'])

        # Permutation test (N=20) for non-baseline conditions
        if condition != MS-D:
            perm = permutation_test(model, train_data, test_data, n_permutations=20)

        store results

aggregate across seeds: mean ± std for AUC, F1
save to results/multi_source_experiments/exp1_full_dataset.pkl
print summary table
```

### CLI

```bash
python transfer_learning/exp1_full_dataset.py \
    --seeds 42 123 456 789 1234 \
    --epochs 100 --batch-size 32 --patience 15 \
    --device cuda
```

### Test criteria

- Script runs without error
- Produces `results/multi_source_experiments/exp1_full_dataset.pkl`
- Each condition has 5 AUC values (one per seed)
- AUC values are in [0, 1]
- Majority baseline AUC ≈ 0.50
- Permutation test p-values are computed

---

## Step 3: Create `transfer_learning/exp2_source_stratified.py`

**Question**: "Does data quality matter more than quantity?"

### Algorithm

```
for seed in [42, 123, 456, 789, 1234]:
    # Train on FULL dataset (same as MS-B, best from exp1)
    train, val, test = load_splits()
    model, result = run_training(train, val, test, pretrained=True,
                                  freeze_backbone=False, lr=1e-4)

    # Evaluate per-source on test set
    test_data = prepare_pyg_data(test)
    gsr_test = [d for d in test_data if d.source == 'soccernet_gsr']
    sc_test = [d for d in test_data if d.source == 'skillcorner']
    dfl_test = [d for d in test_data if d.source == 'dfl']

    for subset_name, subset in [('gsr', gsr_test), ('skillcorner', sc_test), ('dfl', dfl_test)]:
        if len(subset) >= 2 and len(set(d.y.item() for d in subset)) == 2:
            metrics = evaluate(model, DataLoader(subset, batch_size=32), device)
        else:
            metrics = {'auc': float('nan'), 'note': f'only {len(subset)} samples'}

    store per-source metrics

aggregate across seeds
save to results/multi_source_experiments/exp2_source_stratified.pkl

# Known limitation: test split has 0 DFL, 8 SkillCorner, 451 GSR
# Report this prominently in output
```

### Test criteria

- Produces per-source AUC breakdown
- GSR subset has meaningful stats (451 samples)
- SkillCorner/DFL correctly flagged as underpowered

---

## Step 4: Create `transfer_learning/exp3_source_ablation.py`

**Question**: "Does noisy GSR data help or hurt?"

### Conditions

| ID | Train on | Eval on | Description |
|----|----------|---------|-------------|
| S-A | GSR only (2,080 train) | Full test (459) | Noisy but large |
| S-B | DFL+SkillCorner only (76 train) | Full test (459) | Clean but tiny |
| S-C | All combined (2,156 train) | Full test (459) | Everything |

### Algorithm

```
for seed in [42, 123, 456, 789, 1234]:
    train, val, test = load_splits()
    test_data = prepare_pyg_data(test)
    val_data = prepare_pyg_data(val)

    # S-A: GSR only
    gsr_train = [c for c in train if c['source'] == 'soccernet_gsr']
    gsr_val = [c for c in val if c['source'] == 'soccernet_gsr']
    model_a, result_a = run_training(
        prepare_pyg_data(gsr_train), prepare_pyg_data(gsr_val), test_data,
        pretrained=True, freeze_backbone=False, lr=1e-4)

    # S-B: DFL+SkillCorner only
    hq_train = [c for c in train if c['source'] in ('dfl', 'skillcorner')]
    hq_val = [c for c in val if c['source'] in ('dfl', 'skillcorner')]
    # Small dataset: use frozen backbone + smaller head + lower batch size
    model_b, result_b = run_training(
        prepare_pyg_data(hq_train), prepare_pyg_data(hq_val), test_data,
        pretrained=True, freeze_backbone=True, lr=1e-3,
        batch_size=8, head_hidden=32, epochs=50)

    # S-C: All combined (same as MS-B from exp1)
    model_c, result_c = run_training(
        prepare_pyg_data(train), val_data, test_data,
        pretrained=True, freeze_backbone=False, lr=1e-4)

save to results/multi_source_experiments/exp3_source_ablation.pkl
```

### Design note for S-B

DFL+SkillCorner training set is only 76 samples (24 DFL + 52 SkillCorner).
Val is only 28 (4 DFL + 24 SkillCorner). This mirrors the n=57 situation
from Phase 3, so we use frozen backbone + small head (same approach that
worked best in Phase 3). Hyperparams: batch_size=8, head_hidden=32, epochs=50.

### Test criteria

- Three conditions produce valid AUC values
- S-B doesn't crash despite tiny training set (76 samples)
- Results saved correctly

---

## Step 5: Create `transfer_learning/exp4_feature_ablation.py`

**Question**: "Do velocity vectors provide the missing signal, now that we have enough data?"

### Approach

Train two separate models with different feature sets, then compare.

| ID | Features used | Description |
|----|---------------|-------------|
| F-A | All 12 features | Full model (same as MS-B from exp1) |
| F-B | Position only (zero velocity) | Features [2,3,4,5] set to 0 in x tensor |

### Algorithm

```
def zero_velocity_features(data_list: List[Data]) -> List[Data]:
    """Create copies with velocity features zeroed out."""
    new_list = []
    for d in data_list:
        d_new = d.clone()
        d_new.x = d.x.clone()
        d_new.x[:, 2:6] = 0.0  # Zero vx, vy, velocity_mag, velocity_angle
        new_list.append(d_new)
    return new_list

for seed in [42, 123, 456, 789, 1234]:
    train, val, test = load_splits()
    train_data = prepare_pyg_data(train)
    val_data = prepare_pyg_data(val)
    test_data = prepare_pyg_data(test)

    # F-A: Full features (same as MS-B)
    model_a, result_a = run_training(
        train_data, val_data, test_data,
        pretrained=True, freeze_backbone=False, lr=1e-4)

    # F-B: Position only (velocity zeroed)
    train_pos = zero_velocity_features(train_data)
    val_pos = zero_velocity_features(val_data)
    test_pos = zero_velocity_features(test_data)
    model_b, result_b = run_training(
        train_pos, val_pos, test_pos,
        pretrained=True, freeze_backbone=False, lr=1e-4)

    # Bootstrap CI for both
    ci_a = bootstrap_auc_ci(result_a['test_metrics'])
    ci_b = bootstrap_auc_ci(result_b['test_metrics'])

    # Permutation test for both (N=20)
    perm_a = permutation_test(model_a, train_data, test_data)
    perm_b = permutation_test(model_b, train_pos, test_pos)

save to results/multi_source_experiments/exp4_feature_ablation.pkl
print comparison table
```

### Test criteria

- Both conditions train successfully
- Velocity-zeroed model doesn't crash (features still 12-dim, just 4 are 0)
- AUC difference between F-A and F-B is reported with CI
- Permutation test p-values computed for both

---

## Execution order

1. **Step 1**: `multi_source_utils.py` — shared code, no execution
2. **Step 2**: `exp1_full_dataset.py` — run and report numbers
3. **STOP. Report Experiment 1 results before continuing.**
4. Step 3: `exp2_source_stratified.py`
5. Step 4: `exp3_source_ablation.py`
6. Step 5: `exp4_feature_ablation.py`

Per user request: "Start with Experiment 1. Report numbers before moving on."

---

## Hyperparameter summary

| Parameter | Phase 3 (n=57) | Multi-source (n=3,078) | Rationale |
|-----------|----------------|------------------------|-----------|
| head_hidden | 32 | 64 | 54× more data supports larger head |
| batch_size | 8 | 32 | Standard scaling |
| epochs | 50 | 100 | More data needs more epochs |
| patience | 10 | 15 | Allow more exploration |
| lr (frozen) | 1e-4 | 1e-3 | Larger batches can use larger LR |
| lr (unfrozen) | 1e-5 | 1e-4 | Same reasoning |
| seeds | [42,123,456,789,1234] | same | Consistency |

## Risk: Edge features with velocity zeroing

When velocity features [2,3,4,5] are zeroed in node features, edge features
[1] (speed_diff), [4] (velocity_sine_angle), [5] (velocity_cosine_angle) still
contain velocity-derived information. For a clean ablation, we should also zero
edge features [1,4,5] when zeroing node velocity features. This will be handled
in `zero_velocity_features()`.
