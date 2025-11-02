# Baseline Training System Architecture

## Core Components Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASELINE TRAINING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

Data Loading          Model Definitions         Training Script
─────────────         ──────────────────        ────────────────
┌──────────────┐      ┌──────────────────┐     ┌─────────────────┐
│ receiver_    │      │ baselines.py     │     │ train_          │
│ data_loader  │─────▶│                  │────▶│ baselines.py    │
│              │      │ • Random         │     │                 │
│ • Load graphs│      │ • XGBoost        │     │ • Train loop    │
│ • Split data │      │ • MLP            │     │ • Evaluation    │
│ • Batching   │      │                  │     │ • Metrics       │
└──────────────┘      └──────────────────┘     └─────────────────┘
      │                      │                          │
      │                      │                          │
      ▼                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RESULTS                                  │
│  • JSON files (model metrics)                                   │
│  • Saved models (.pt files)                                     │
│  • Report visualizations                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Loading (`src/data/receiver_data_loader.py`)

**File**: `src/data/receiver_data_loader.py` (15KB)

### Key Classes:

```python
class ReceiverCornerDataset(Dataset):
    """PyTorch Dataset for corner kick graphs with receiver labels."""

    def __init__(self, graphs, mask_velocities=True):
        # Load graphs, filter for receiver labels
        # Mask velocities for StatsBomb data (not available)

    def __getitem__(self, idx):
        # Returns: PyTorch Geometric Data object
        # - graph.x: node features [num_nodes, 14]
        # - graph.edge_index: adjacency [2, num_edges]
        # - graph.receiver_label: target receiver [1]
        # - graph.shot_label: shot/goal binary [1]
```

### Main Function:

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
    Returns:
        dataset: Full dataset
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
```

**What it does:**
- Loads graph dataset from pickle file
- Filters for corners with receiver labels (60% have labels)
- Splits data: 70% train, 15% val, 15% test
- Creates PyTorch DataLoaders with batching
- Masks velocity features (vx, vy) for StatsBomb data

---

## 2. Model Definitions (`src/models/baselines.py`)

**File**: `src/models/baselines.py` (35KB)

### Three Baseline Models:

#### A. Random Baseline
```python
class RandomReceiverBaseline:
    """Random prediction for sanity checking."""

    def predict_receiver(self, graph_x, batch_indices):
        # Returns: Random uniform distribution over 22 players
        return torch.rand(batch_size, 22)

    def predict_shot(self, graph_x, batch_indices):
        # Returns: Random binary prediction
        return torch.rand(batch_size, 1)
```

**Purpose**: Validates that evaluation metrics work correctly. Should get ~13.6% Top-3 (3/22 players).

---

#### B. XGBoost Baseline
```python
class XGBoostReceiverBaseline:
    """Gradient boosted trees with engineered features."""

    def __init__(self, max_depth=6, n_estimators=500, learning_rate=0.05):
        self.receiver_model = xgb.XGBClassifier(...)
        self.shot_model = xgb.XGBClassifier(...)

    def _prepare_features(self, graph_data):
        """Extract features from graph."""
        # For each player node:
        # - Position (x, y)
        # - Distance to goal
        # - Distance to ball
        # - Angles
        # - Local density
        #
        # Aggregate statistics:
        # - Mean/std of features
        # - Min/max distances
        # - Team formations

    def fit(self, train_loader):
        # Train two XGBoost models:
        # 1. Receiver classifier (22 classes)
        # 2. Shot classifier (binary)
```

**Key Features**:
- Engineered features from graph structure
- Two separate XGBoost models (receiver + shot)
- Hand-crafted features: positions, distances, angles, densities
- No training loop needed (single pass fit)

**Best Results**: 89.3% Top-3 accuracy

---

#### C. MLP Baseline
```python
class MLPReceiverBaseline(nn.Module):
    """Multi-layer perceptron with dual-task learning."""

    def __init__(
        self,
        input_dim: int = 14,      # Node features
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.25,
        num_receivers: int = 22,   # Output classes
        num_shot: int = 1          # Binary shot prediction
    ):
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.receiver_head = nn.Linear(hidden2, num_receivers)
        self.shot_head = nn.Linear(hidden2, num_shot)

    def forward(self, graph_x, batch_indices):
        """
        Args:
            graph_x: Node features [total_nodes, 14]
            batch_indices: Which graph each node belongs to

        Returns:
            receiver_logits: [batch_size, 22]
            shot_logits: [batch_size, 1]
        """
        # 1. Pool node features to graph-level (mean pooling)
        graph_features = global_mean_pool(graph_x, batch_indices)

        # 2. Shared encoding
        encoded = self.encoder(graph_features)

        # 3. Task-specific predictions
        receiver_logits = self.receiver_head(encoded)
        shot_logits = self.shot_head(encoded)

        return receiver_logits, shot_logits
```

**Architecture**:
```
Input [14-dim node features]
         ↓
   Mean Pooling (per graph)
         ↓
   [batch_size × 14]
         ↓
   Linear(14 → 512) + ReLU + Dropout(0.25)
         ↓
   Linear(512 → 256) + ReLU + Dropout(0.25)
         ↓
   ┌──────────────────┐
   ↓                  ↓
Receiver Head     Shot Head
Linear(256→22)    Linear(256→1)
   ↓                  ↓
[22 classes]      [binary]
```

**Training**: 20,000 gradient steps with Adam optimizer

**Results**: 66.7% Top-3 accuracy

---

## 3. Training Script (`scripts/training/train_baselines.py`)

**File**: `scripts/training/train_baselines.py` (17KB)

### Main Training Flow:

```python
def main(args):
    # 1. Setup
    device = torch.device(f'cuda:{args.gpu_id}')
    set_seed(args.seed)

    # 2. Load data
    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=args.data_path,
        batch_size=args.batch_size,
        mask_velocities=True
    )

    # 3. Initialize model
    if args.model == 'random':
        model = RandomReceiverBaseline()
    elif args.model == 'xgboost':
        model = XGBoostReceiverBaseline(...)
        model.fit(train_loader, val_loader)  # Train XGBoost
    elif args.model == 'mlp':
        model = MLPReceiverBaseline(...).to(device)
        train_mlp(model, train_loader, val_loader, ...)  # Training loop

    # 4. Evaluate on test set
    test_receiver_metrics = evaluate_receiver_model(model, test_loader, device)
    test_shot_metrics = evaluate_shot_model(model, test_loader, device)

    # 5. Save results
    results = {
        'model': args.model,
        'test_receiver_metrics': test_receiver_metrics,
        'test_shot_metrics': test_shot_metrics,
        'args': vars(args)
    }
    save_results(results, f'results/baselines/{args.model}_results.json')
```

### Key Functions:

#### A. Training Loop (MLP only)
```python
def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_steps: int = 20000,
    lr: float = 0.0005,
    shot_weight: float = 1.5,
    device: torch.device = None
):
    """Train MLP with dual-task loss."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    receiver_criterion = nn.CrossEntropyLoss()
    shot_criterion = nn.BCEWithLogitsLoss()

    for step in range(num_steps):
        # Forward pass
        receiver_logits, shot_logits = model(batch.x, batch.batch)

        # Dual-task loss
        receiver_loss = receiver_criterion(receiver_logits, batch.receiver_label)
        shot_loss = shot_criterion(shot_logits, batch.shot_label)
        total_loss = receiver_loss + shot_weight * shot_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Validation every 1000 steps
        if step % 1000 == 0:
            val_metrics = evaluate(model, val_loader, device)
            log_metrics(step, val_metrics)
```

---

#### B. Evaluation Functions
```python
def evaluate_receiver_model(model, test_loader, device):
    """Compute Top-K accuracy for receiver prediction."""
    all_logits = []
    all_labels = []

    for batch in test_loader:
        logits = model.predict_receiver(batch.x, batch.batch)
        all_logits.append(logits)
        all_labels.append(batch.receiver_label)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Compute Top-1, Top-3, Top-5 accuracy
    metrics = compute_topk_accuracy(all_logits, all_labels, k_values=[1, 3, 5])
    return metrics


def evaluate_shot_model(model, test_loader, device):
    """Compute shot prediction metrics."""
    all_probs = []
    all_labels = []

    for batch in test_loader:
        shot_logits = model.predict_shot(batch.x, batch.batch)
        probs = torch.sigmoid(shot_logits)
        all_probs.append(probs)
        all_labels.append(batch.shot_label)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # Compute F1, Precision, Recall, AUROC, AUPRC
    metrics = compute_shot_metrics(all_probs, all_labels, threshold=0.5)
    return metrics
```

---

#### C. Metrics Computation
```python
def compute_topk_accuracy(logits, labels, k_values=[1, 3, 5]):
    """Top-K accuracy for multi-class prediction."""
    probs = F.softmax(logits, dim=1)
    results = {}

    for k in k_values:
        if k == 1:
            pred = probs.argmax(dim=1)
            acc = (pred == labels).float().mean()
        else:
            topk = probs.topk(k, dim=1)[1]
            acc = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f'top{k}'] = acc.item()

    return results


def compute_shot_metrics(probs, labels, threshold=0.5):
    """Binary classification metrics for shot prediction."""
    preds = (probs >= threshold).float()

    # Confusion matrix
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # AUROC and AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(labels.cpu(), probs.cpu())
    auprc = average_precision_score(labels.cpu(), probs.cpu())

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
        'auprc': auprc
    }
```

---

## 4. SLURM Execution

**Scripts**: `scripts/slurm/train_baselines_*.sh`

```bash
#!/bin/bash
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/baselines_a100_%j.out
#SBATCH --error=logs/baselines_a100_%j.err

conda activate robo
cd /home/mseo/CornerTactics

# Train all three baselines sequentially
python scripts/training/train_baselines.py --model random --gpu-type a100
python scripts/training/train_baselines.py --model xgboost --gpu-type a100
python scripts/training/train_baselines.py --model mlp --gpu-type a100 --mlp-steps 20000
```

**Execution**:
```bash
sbatch scripts/slurm/train_baselines_a100.sh  # A100 GPU
sbatch scripts/slurm/train_baselines_v100.sh  # V100 GPU
```

---

## 5. Results Generation

**Script**: `scripts/analysis/generate_baseline_report.py`

```python
def main():
    # 1. Load all JSON results
    results = load_results()  # From results/baselines/*.json

    # 2. Generate comparison table
    df = create_comparison_table(results)
    df.to_csv('baseline_comparison.csv')

    # 3. Generate LaTeX table
    latex = create_latex_table(df)

    # 4. Generate visualizations
    plot_receiver_prediction_comparison(results)  # Bar chart
    plot_shot_prediction_comparison(results)      # Dual bar charts
    plot_combined_performance_radar(results)      # Radar chart

    # 5. Generate summary statistics
    generate_summary_statistics(results)
```

---

## Data Flow Diagram

```
Pickle File                    DataLoader              Model                Results
────────────                   ──────────              ─────                ───────

statsbomb_                     ReceiverCorner          [Random/              JSON
temporal_                      Dataset                 XGBoost/              Files
augmented_                          │                  MLP]                   │
with_receiver.pkl                   │                    │                    │
     │                              │                    │                    │
     │  1. Load                     │                    │                    │
     ├──────────────────────────────┤                    │                    │
     │                              │                    │                    │
     │  2. Filter (receiver labels) │                    │                    │
     ├──────────────────────────────┤                    │                    │
     │                              │                    │                    │
     │  3. Split (70/15/15)         │                    │                    │
     ├──────────────────────────────┤                    │                    │
     │                              │                    │                    │
     │  4. Batch (size=128)         │   5. Train/Eval    │                    │
     │                              ├────────────────────┤                    │
     │                              │                    │                    │
     │                              │   6. Predict       │                    │
     │                              ├────────────────────┤                    │
     │                              │                    │                    │
     │                              │   7. Compute       │   8. Save          │
     │                              │      Metrics       ├────────────────────┤
     │                              │                    │                    │
     │                              │                    │                    ▼
     │                              │                    │           random_results.json
     │                              │                    │           xgboost_results.json
     │                              │                    │           mlp_results.json
```

---

## Summary: Core Components

| Component | File | Size | Purpose |
|-----------|------|------|---------|
| **Data Loader** | `src/data/receiver_data_loader.py` | 15KB | Load graphs, split data, create batches |
| **Models** | `src/models/baselines.py` | 35KB | Define Random, XGBoost, MLP baselines |
| **Training** | `scripts/training/train_baselines.py` | 17KB | Train models, evaluate, save results |
| **SLURM** | `scripts/slurm/train_baselines_*.sh` | ~1KB | Submit GPU jobs |
| **Reporting** | `scripts/analysis/generate_baseline_report.py` | ~10KB | Generate tables and visualizations |

**Total Code**: ~77KB across 5 core files

---

## Quick Start

```bash
# 1. Train all baselines (A100 GPU)
sbatch scripts/slurm/train_baselines_a100.sh

# 2. Check results
cat results/baselines/xgboost_results_a100.json

# 3. Generate report
python scripts/analysis/generate_baseline_report.py

# 4. View visualizations
ls results/baselines/report/
```

---

## Key Design Decisions

1. **Dual-Task Learning**: All models predict both receiver (22-class) and shot (binary)
2. **Graph Pooling**: MLP uses mean pooling to convert node features → graph features
3. **Engineered Features**: XGBoost uses hand-crafted features (positions, distances, angles)
4. **Learned Features**: MLP learns representations end-to-end
5. **Evaluation**: Top-K accuracy (k=1,3,5) for receiver, F1/AUROC for shot prediction
6. **Production Ready**: No fallback code, CUDA device handling, proper error handling
