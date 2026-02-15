"""Baseline 5: MLP on per-player features (no graph structure).

Flatten all 22 players × 13 features → 286-dim vector → MLP → P(shot).
Tests whether graph structure adds value beyond having the right features.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from corner_prediction.config import (
    BATCH_SIZE,
    SHOT_DROPOUT,
    SHOT_EPOCHS,
    SHOT_LR,
    SHOT_PATIENCE,
    SHOT_POS_WEIGHT,
    SHOT_WEIGHT_DECAY,
)
from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.evaluate import (
    compute_receiver_metrics,
    compute_shot_metrics,
)

logger = logging.getLogger(__name__)

N_PLAYERS = 22
NODE_FEATURES = 13
FLAT_DIM = N_PLAYERS * NODE_FEATURES  # 286


class ShotMLP(nn.Module):
    """Simple MLP for shot prediction from flattened player features."""

    def __init__(
        self,
        input_dim: int = FLAT_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Flattened features [B, input_dim].

        Returns:
            Logits [B, 1].
        """
        return self.net(x)


def _flatten_graph(graph) -> np.ndarray:
    """Flatten a graph's node features into a single vector.

    Nodes are already in a consistent order from graph construction.
    Output shape: [N_PLAYERS * NODE_FEATURES] = [286].
    """
    x = graph.x.numpy()  # [n_nodes, 13]
    n_nodes = x.shape[0]

    if n_nodes < N_PLAYERS:
        # Pad with zeros if fewer players
        pad = np.zeros((N_PLAYERS - n_nodes, NODE_FEATURES))
        x = np.vstack([x, pad])
    elif n_nodes > N_PLAYERS:
        x = x[:N_PLAYERS]

    return x.flatten()


def _build_tensors(data_list: List) -> tuple:
    """Build flat feature and label tensors from a list of graphs."""
    X = np.array([_flatten_graph(g) for g in data_list], dtype=np.float32)
    y = np.array([g.shot_label for g in data_list], dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def _mlp_fold(
    train_data: List,
    val_data: List,
    test_data: List,
    seed: int,
    device: torch.device,
    hidden_dim: int = 64,
    dropout: float = SHOT_DROPOUT,
    lr: float = SHOT_LR,
    epochs: int = SHOT_EPOCHS,
    patience: int = SHOT_PATIENCE,
    weight_decay: float = SHOT_WEIGHT_DECAY,
    pos_weight: float = SHOT_POS_WEIGHT,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Train and evaluate MLP on one fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = _build_tensors(train_data)
    X_val, y_val = _build_tensors(val_data)
    X_test, y_test = _build_tensors(test_data)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ShotMLP(input_dim=FLAT_DIM, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor([pos_weight], device=device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, y_batch, pos_weight=pw,
            )
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, y_val.to(device), pos_weight=pw,
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze(-1)
        probs = torch.sigmoid(test_logits).cpu().numpy()

    y_test_np = y_test.numpy()
    n_samples = len(y_test_np)

    # AUC
    if len(np.unique(y_test_np)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_test_np, probs)

    # F1 at optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_test_np, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_05 = (probs >= 0.5).astype(int)
    accuracy = float((preds_05 == y_test_np).mean())

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "accuracy": accuracy,
        "probs": probs.tolist(),
        "labels": y_test_np.tolist(),
        "n_samples": n_samples,
        "n_positive": int(y_test_np.sum()),
    }


def mlp_baseline_lomo(
    dataset,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run MLP baseline with LOMO cross-validation.

    Shot prediction only — no receiver prediction.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match_ids = get_match_ids(dataset)
    fold_results = []

    dummy_receiver = {
        "top1_acc": 0.0, "top3_acc": 0.0, "n_labeled": 0, "per_graph": [],
    }

    for fold_idx, held_out in enumerate(match_ids):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        if verbose:
            print(f"\n--- MLP Baseline Fold {fold_idx + 1}/{len(match_ids)}: "
                  f"held_out={held_out} ---")

        train_data, test_data = lomo_split(dataset, held_out)
        if not test_data:
            continue

        # Inner validation split (same as lomo_cv)
        val_match = match_ids[(fold_idx + 1) % len(match_ids)]
        inner_train = [g for g in train_data if str(g.match_id) != val_match]
        val_data = [g for g in train_data if str(g.match_id) == val_match]

        if not val_data:
            n_val = max(1, len(train_data) // 5)
            val_data = train_data[:n_val]
            inner_train = train_data[n_val:]

        shot = _mlp_fold(inner_train, val_data, test_data, seed + fold_idx, device)

        fold_result = {
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(inner_train),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "receiver": dummy_receiver,
            "shot_oracle": shot,
            "shot_predicted": shot,
            "shot_unconditional": shot,
        }
        fold_results.append(fold_result)

        if verbose:
            n_train_shots = sum(1 for g in inner_train if g.shot_label == 1)
            print(f"  train={len(inner_train)} ({n_train_shots} shots), "
                  f"val={len(val_data)}, test={len(test_data)}")
            print(f"  Shot: AUC={shot['auc']:.3f}, F1={shot['f1']:.3f}")

    agg_shot = compute_shot_metrics(fold_results, "oracle")

    dummy_agg_receiver = {
        "top1_mean": 0.0, "top1_std": 0.0, "top3_mean": 0.0, "top3_std": 0.0,
        "n_folds": 0, "per_fold_top1": [], "per_fold_top3": [],
    }

    results = {
        "config": {
            "baseline": "mlp",
            "seed": seed,
            "input_dim": FLAT_DIM,
            "hidden_dim": 64,
            "n_folds": len(fold_results),
        },
        "per_fold": fold_results,
        "aggregated": {
            "receiver": dummy_agg_receiver,
            "shot_oracle": agg_shot,
            "shot_predicted": agg_shot,
            "shot_unconditional": agg_shot,
        },
    }

    if verbose:
        _print_mlp_results(results)

    return results


def _print_mlp_results(results: Dict) -> None:
    s = results["aggregated"]["shot_oracle"]
    print(f"\n{'=' * 60}")
    print("MLP Baseline Results (Shot Prediction Only)")
    print(f"{'=' * 60}")
    print(f"Architecture: Linear({FLAT_DIM}, 64) → ReLU → Dropout → Linear(64, 1)")
    print(f"Shot AUC: {s['auc_mean']:.3f} +/- {s['auc_std']:.3f}")
    print(f"Shot F1:  {s['f1_mean']:.3f} +/- {s['f1_std']:.3f}")
    print(f"{'=' * 60}")
