#!/usr/bin/env python3
"""
Shared utilities for multi-source corner kick prediction experiments.
=====================================================================

Provides model definitions, data loading, training loops, evaluation,
and statistical testing for experiments on the 3,078-corner multi-source
dataset (DFL + SkillCorner + SoccerNet GSR).

Adapted from phase3_transfer_learning.py with hyperparameters scaled
for the larger dataset.
"""

import copy
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-sparse.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "multi_source_experiments"

# ---------------------------------------------------------------------------
# Feature schema (USSF 12-dim node features)
# ---------------------------------------------------------------------------
FEATURE_INDICES = {
    "x": 0,
    "y": 1,
    "vx": 2,
    "vy": 3,
    "velocity_mag": 4,
    "velocity_angle": 5,
    "dist_goal": 6,
    "angle_goal": 7,
    "dist_ball": 8,
    "angle_ball": 9,
    "attacking_team_flag": 10,
    "potential_receiver": 11,
}
VELOCITY_NODE_FEATURES = [2, 3, 4, 5]
VELOCITY_EDGE_FEATURES = [1, 4, 5]  # speed_diff, vel_sin, vel_cos
POSITION_FEATURES = [0, 1, 6, 7, 8, 9]


# ===================================================================
# Model
# ===================================================================

class TransferGNN(nn.Module):
    """CrystalConv GNN with optional pretrained backbone.

    Architecture matches USSF CounterattackGNN backbone so we can load
    pretrained weights directly.
    """

    def __init__(
        self,
        node_features: int = 12,
        edge_features: int = 6,
        hidden_channels: int = 128,
        num_conv_layers: int = 3,
        head_hidden: int = 64,
        head_dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.freeze_backbone = freeze_backbone

        # Backbone ---------------------------------------------------------
        self.conv1 = CGConv(channels=node_features, dim=edge_features, aggr="add")
        self.lin_in = nn.Linear(node_features, hidden_channels)
        self.convs = nn.ModuleList(
            [
                CGConv(channels=hidden_channels, dim=edge_features, aggr="add")
                for _ in range(num_conv_layers - 1)
            ]
        )

        # Head -------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        if freeze_backbone:
            self._freeze_backbone()

    # -- backbone control --------------------------------------------------

    def _freeze_backbone(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.lin_in.parameters():
            param.requires_grad = True
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True

    def load_pretrained_backbone(self, backbone_state: Dict):
        self.conv1.load_state_dict(backbone_state["conv1"])
        self.lin_in.load_state_dict(backbone_state["lin_in"])
        for i, conv_state in enumerate(backbone_state["convs"]):
            self.convs[i].load_state_dict(conv_state)

    # -- forward -----------------------------------------------------------

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.lin_in(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        return self.head(x)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===================================================================
# Data loading
# ===================================================================

def load_splits() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load pre-built multi-source train/val/test splits."""
    train = pickle.load(open(DATA_DIR / "multi_source_corners_dense_train.pkl", "rb"))
    val = pickle.load(open(DATA_DIR / "multi_source_corners_dense_val.pkl", "rb"))
    test = pickle.load(open(DATA_DIR / "multi_source_corners_dense_test.pkl", "rb"))
    return train, val, test


def prepare_pyg_data(corners: List[Dict]) -> List[Data]:
    """Convert corner sample dicts to PyG Data objects."""
    data_list = []
    for sample in corners:
        graph = sample["graphs"][0]
        label = float(sample["labels"]["shot_binary"])
        pyg_data = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=torch.tensor([label], dtype=torch.float32),
        )
        pyg_data.match_id = sample["match_id"]
        pyg_data.corner_time = sample.get("corner_time", 0.0)
        pyg_data.source = sample.get("source", "unknown")
        data_list.append(pyg_data)
    return data_list


def zero_velocity_features(data_list: List[Data]) -> List[Data]:
    """Return copies with velocity features zeroed in both nodes and edges."""
    new_list = []
    for d in data_list:
        d_new = d.clone()
        d_new.x = d.x.clone()
        d_new.x[:, VELOCITY_NODE_FEATURES] = 0.0
        if d_new.edge_attr is not None:
            d_new.edge_attr = d.edge_attr.clone()
            d_new.edge_attr[:, VELOCITY_EDGE_FEATURES] = 0.0
        new_list.append(d_new)
    return new_list


# ===================================================================
# kNN graph construction
# ===================================================================

def _compute_edge_features_batch(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute 6-dim edge features for all edges (vectorized).

    Matches graph_converter._compute_edge_features exactly:
        [dist_norm, speed_diff, pos_sine, pos_cosine, vel_sine, vel_cosine]
    """
    src = x[edge_index[0]]  # [n_edges, 12]
    dst = x[edge_index[1]]  # [n_edges, 12]

    xi, yi = src[:, 0], src[:, 1]
    xj, yj = dst[:, 0], dst[:, 1]
    vxi, vyi = src[:, 2], src[:, 3]
    vxj, vyj = dst[:, 2], dst[:, 3]
    vel_mag_i = src[:, 4]
    vel_mag_j = dst[:, 4]

    # 0: distance (normalized by sqrt(2) = max possible in [0,1]^2)
    dx = xj - xi
    dy = yj - yi
    dist = torch.sqrt(dx * dx + dy * dy)
    dist_norm = torch.clamp(dist / np.sqrt(2), 0.0, 1.0)

    # 1: speed difference
    speed_diff = vel_mag_j - vel_mag_i

    # 2-3: positional angle sin/cos
    pos_angle = torch.atan2(dy, dx)
    pos_sine = (torch.sin(pos_angle) + 1) / 2
    pos_cosine = (torch.cos(pos_angle) + 1) / 2

    # 4-5: velocity angle sin/cos
    dot = vxi * vxj + vyi * vyj
    cross = vxi * vyj - vyi * vxj
    vel_angle = torch.atan2(cross, dot)
    vel_sine = (torch.sin(vel_angle) + 1) / 2
    vel_cosine = (torch.cos(vel_angle) + 1) / 2

    return torch.stack([dist_norm, speed_diff, pos_sine, pos_cosine,
                        vel_sine, vel_cosine], dim=1).float()


def rebuild_knn_edges(data_list: List[Data], k: int = 5) -> List[Data]:
    """Replace dense edges with kNN (Euclidean distance on x,y positions).

    For each graph, finds k nearest neighbors per node and recomputes
    edge features. When n_nodes <= k+1, degenerates to dense (correct).
    """
    from scipy.spatial.distance import cdist

    new_list = []
    for d in data_list:
        n_nodes = d.x.shape[0]
        pos = d.x[:, :2].numpy()  # normalized x, y
        effective_k = min(k, n_nodes - 1)

        # Pairwise distance matrix
        dist_matrix = cdist(pos, pos)
        np.fill_diagonal(dist_matrix, np.inf)

        # kNN: for each node, pick `effective_k` nearest
        src, dst = [], []
        for i in range(n_nodes):
            neighbors = np.argsort(dist_matrix[i])[:effective_k]
            for j in neighbors:
                src.append(i)
                dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = _compute_edge_features_batch(d.x, edge_index)

        d_new = Data(
            x=d.x.clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=d.y.clone() if d.y is not None else None,
        )
        # Preserve metadata
        if hasattr(d, "match_id"):
            d_new.match_id = d.match_id
        if hasattr(d, "corner_time"):
            d_new.corner_time = d.corner_time
        if hasattr(d, "source"):
            d_new.source = d.source
        new_list.append(d_new)

    return new_list


# ===================================================================
# Training utilities
# ===================================================================

def compute_class_weights(labels: List[float]) -> torch.Tensor:
    """Inverse-frequency class weights for binary classification."""
    labels = np.array(labels)
    n_samples = len(labels)
    n_pos = labels.sum()
    n_neg = n_samples - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])
    w_neg = n_samples / (2 * n_neg)
    w_pos = n_samples / (2 * n_pos)
    return torch.tensor([w_neg, w_pos])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.squeeze(), pos_weight=pos_weight.to(device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return total_loss / n_samples if n_samples > 0 else 0.0


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model. Returns dict with auc, accuracy, precision, recall, f1."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.squeeze()
            )
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs])
            labels = batch.y.squeeze().cpu().numpy()
            if labels.ndim == 0:
                labels = np.array([labels])

            all_probs.extend(probs.tolist())
            all_preds.extend((probs > 0.5).astype(int).tolist())
            all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return {
        "loss": total_loss / n_samples if n_samples > 0 else 0.0,
        "auc": auc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": n_samples,
        "n_positive": int(all_labels.sum()),
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "labels": all_labels.tolist(),
    }


# ===================================================================
# Model factory
# ===================================================================

def create_model(
    pretrained: bool = False,
    freeze_backbone: bool = False,
    backbone_path: str = "ussf_backbone_dense.pt",
    head_hidden: int = 64,
    head_dropout: float = 0.3,
    device: str = "cpu",
) -> TransferGNN:
    """Create TransferGNN, optionally loading pretrained backbone."""
    model = TransferGNN(
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)
    if pretrained:
        path = WEIGHTS_DIR / backbone_path
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_pretrained_backbone(state)
        else:
            print(f"  WARNING: {path} not found, using random init")
    return model


# ===================================================================
# Training runner
# ===================================================================

def run_training(
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    *,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    backbone_path: str = "ussf_backbone_dense.pt",
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    head_hidden: int = 64,
    head_dropout: float = 0.3,
    device: Optional[str] = None,
    seed: int = 42,
    label: str = "experiment",
    verbose: bool = True,
) -> Tuple[TransferGNN, Dict]:
    """Train a model end-to-end with early stopping.

    Returns (trained_model, results_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    if verbose:
        print(f"\n{'='*60}")
        print(f"{label}: pretrained={pretrained}, frozen={freeze_backbone}, lr={lr}")
        print(f"{'='*60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train_labels = [d.y.item() for d in train_data]
    class_weights = compute_class_weights(train_labels)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]])

    if verbose:
        print(f"  Train: {len(train_data)} ({sum(train_labels):.0f} pos)")
        print(f"  Val:   {len(val_data)}")
        print(f"  Test:  {len(test_data)}")
        print(f"  pos_weight: {pos_weight.item():.3f}")

    model = create_model(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        backbone_path=backbone_path,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        device=device,
    )

    if verbose:
        print(
            f"  Params: {model.count_trainable_params():,} trainable "
            f"/ {model.count_total_params():,} total"
        )

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "best_epoch": 0,
        "best_val_auc": 0.0,
    }
    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, device_obj, pos_weight)
        v_metrics = evaluate(model, val_loader, device_obj)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_metrics["loss"])
        history["val_auc"].append(v_metrics["auc"])

        improved = v_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = v_metrics["auc"]
            best_model_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_auc"] = best_val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1 or improved):
            mark = " *" if improved else ""
            print(
                f"  Epoch {epoch:3d}: train_loss={t_loss:.4f} "
                f"val_auc={v_metrics['auc']:.4f}{mark}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device_obj)
    train_metrics = evaluate(model, train_loader, device_obj)

    if verbose:
        print(f"  Best val AUC: {history['best_val_auc']:.4f} (epoch {history['best_epoch']})")
        print(f"  Test AUC:  {test_metrics['auc']:.4f}")
        print(f"  Test F1:   {test_metrics['f1']:.4f}")

    result = {
        "label": label,
        "config": {
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "head_hidden": head_hidden,
            "seed": seed,
        },
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": evaluate(model, val_loader, device_obj),
        "test_metrics": test_metrics,
    }

    return model, result


# ===================================================================
# Baselines and statistical tests
# ===================================================================

def majority_baseline(
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
) -> Dict:
    """Majority-class baseline (always predict most common class)."""
    train_labels = np.array([d.y.item() for d in train_data])
    majority_class = 1.0 if train_labels.mean() > 0.5 else 0.0
    prob = float(train_labels.mean())

    results = {}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        labels = np.array([d.y.item() for d in data])
        preds = np.full_like(labels, majority_class)
        probs = np.full_like(labels, prob)

        if len(np.unique(labels)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(labels, probs)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        results[f"{name}_metrics"] = {
            "auc": auc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": len(labels),
            "n_positive": int(labels.sum()),
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "labels": labels.tolist(),
        }

    return {
        "label": "majority_baseline",
        "config": {"majority_class": int(majority_class), "positive_rate": prob},
        "train_metrics": results["train_metrics"],
        "val_metrics": results["val_metrics"],
        "test_metrics": results["test_metrics"],
    }


def bootstrap_auc_ci(
    metrics: Dict,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap 95% CI for AUC from evaluate() output."""
    rng = np.random.RandomState(seed)
    labels = np.array(metrics.get("labels", []))
    probs = np.array(metrics.get("probabilities", []))
    if len(labels) < 5:
        return {}

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))

    if not aucs:
        return {}

    alpha = 1 - ci_level
    return {
        "auc_lower": float(np.percentile(aucs, alpha / 2 * 100)),
        "auc_upper": float(np.percentile(aucs, (1 - alpha / 2) * 100)),
        "auc_median": float(np.median(aucs)),
        "n_valid_bootstraps": len(aucs),
    }


def permutation_test(
    backbone_state: Dict,
    train_data: List[Data],
    test_data: List[Data],
    *,
    n_permutations: int = 20,
    epochs: int = 30,
    batch_size: int = 32,
    head_hidden: int = 64,
    device: str = "cpu",
    seed: int = 42,
    observed_auc: float = 0.5,
) -> Dict:
    """Label permutation test.

    Trains a fresh head on shuffled labels for each permutation (backbone
    frozen). Compares null AUC distribution to observed AUC.

    Returns dict with null_aucs, p_value, observed_auc.
    """
    rng = np.random.RandomState(seed)
    device_obj = torch.device(device)
    null_aucs = []

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for i in range(n_permutations):
        perm_seed = rng.randint(0, 2**31)
        torch.manual_seed(perm_seed)

        # Shuffle train labels
        shuffled_train = []
        perm_labels = rng.permutation([d.y.item() for d in train_data])
        for j, d in enumerate(train_data):
            d_new = d.clone()
            d_new.y = torch.tensor([perm_labels[j]], dtype=torch.float32)
            shuffled_train.append(d_new)

        # Build model with frozen backbone
        model = TransferGNN(
            head_hidden=head_hidden, freeze_backbone=True
        ).to(device_obj)
        model.load_pretrained_backbone(backbone_state)

        # Train head only
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        train_loader = DataLoader(
            shuffled_train, batch_size=batch_size, shuffle=True
        )
        train_labels = [d.y.item() for d in shuffled_train]
        cw = compute_class_weights(train_labels)
        pw = torch.tensor([cw[1] / cw[0]])

        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, device_obj, pw)

        # Evaluate on real test labels
        metrics = evaluate(model, test_loader, device_obj)
        null_aucs.append(metrics["auc"])

    null_aucs = np.array(null_aucs)
    # p-value: fraction of null AUCs >= observed (one-sided)
    p_value = float((np.sum(null_aucs >= observed_auc) + 1) / (n_permutations + 1))

    return {
        "observed_auc": observed_auc,
        "null_aucs": null_aucs.tolist(),
        "null_auc_mean": float(null_aucs.mean()),
        "null_auc_std": float(null_aucs.std()),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ===================================================================
# I/O
# ===================================================================

def save_results(results: Dict, filename: str) -> Path:
    """Save results dict to pickle in RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved: {path}")
    return path


def print_summary_table(aggregated: Dict[str, Dict], title: str = "Results"):
    """Pretty-print aggregated multi-seed results."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Condition':<20} {'AUC':>12} {'F1':>12} {'Acc':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for cond, stats in sorted(aggregated.items()):
        auc_str = f"{stats['test_auc_mean']:.3f}±{stats['test_auc_std']:.3f}"
        f1_str = f"{stats.get('test_f1_mean', 0):.3f}±{stats.get('test_f1_std', 0):.3f}"
        acc_str = f"{stats.get('test_acc_mean', 0):.3f}±{stats.get('test_acc_std', 0):.3f}"
        print(f"  {cond:<20} {auc_str:>12} {f1_str:>12} {acc_str:>12}")
    print(f"{'='*70}")
