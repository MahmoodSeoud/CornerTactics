"""Core training functions for the two-stage corner kick prediction model.

Provides epoch-level training and evaluation for both stages:
    Stage 1: Receiver prediction (per-node classification)
    Stage 2: Shot prediction (graph-level binary classification)
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from corner_prediction.models import (
    CornerBackbone,
    ReceiverHead,
    ShotHead,
    TwoStageModel,
    masked_softmax,
    receiver_loss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(
    backbone_mode: str = "pretrained",
    pretrained_path: Optional[str] = None,
    freeze: bool = True,
    receiver_hidden: int = 64,
    receiver_dropout: float = 0.3,
    shot_hidden: int = 32,
    shot_dropout: float = 0.3,
    graph_feature_dim: int = 1,
) -> TwoStageModel:
    """Construct the full two-stage model.

    Args:
        backbone_mode: "pretrained" or "scratch".
        pretrained_path: Path to USSF backbone weights. Ignored for scratch.
        freeze: Freeze backbone conv layers (only for pretrained mode).
        receiver_hidden: Hidden dim for receiver head.
        receiver_dropout: Dropout for receiver head.
        shot_hidden: Hidden dim for shot head.
        shot_dropout: Dropout for shot head.
        graph_feature_dim: Number of graph-level features (1 for corner_side).

    Returns:
        TwoStageModel ready for training.
    """
    backbone = CornerBackbone(
        mode=backbone_mode,
        node_features=14,  # 13 graph features + 1 receiver indicator
        edge_features=4,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        freeze=freeze,
    )

    receiver_head = ReceiverHead(
        input_dim=backbone.output_dim,
        hidden_dim=receiver_hidden,
        dropout=receiver_dropout,
    )

    shot_head = ShotHead(
        input_dim=backbone.output_dim,
        graph_feature_dim=graph_feature_dim,
        hidden_dim=shot_hidden,
        dropout=shot_dropout,
    )

    return TwoStageModel(backbone, receiver_head, shot_head)


# ---------------------------------------------------------------------------
# Stage 1: Receiver training
# ---------------------------------------------------------------------------


def train_receiver_epoch(
    model: TwoStageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train receiver head for one epoch.

    Only graphs with has_receiver_label=True contribute to the loss.

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Augment with zeros for receiver indicator (Stage 1)
        x_aug = TwoStageModel._augment_with_receiver(batch.x)

        # Backbone → per-node embeddings
        node_emb = model.backbone(x_aug, batch.edge_index, batch.edge_attr)

        # Receiver head → logits
        logits = model.receiver_head(node_emb)

        # Loss over masked candidates (only labeled graphs)
        loss = receiver_loss(
            logits,
            batch.receiver_label,
            batch.receiver_mask,
            batch.batch,
            batch.has_receiver_label,
        )

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Stage 2: Shot training
# ---------------------------------------------------------------------------


def _get_receiver_indicator(
    model: TwoStageModel,
    batch,
    receiver_mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build receiver indicator tensor for Stage 2.

    Args:
        receiver_mode: "oracle" | "predicted" | "none"

    Returns:
        [N] tensor with 1.0 at receiver node, or None for unconditional.
    """
    if receiver_mode == "none":
        return None
    elif receiver_mode == "oracle":
        return batch.receiver_label.to(device)
    elif receiver_mode == "predicted":
        with torch.no_grad():
            probs = model.predict_receiver(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.receiver_mask, batch.batch,
            )
        # Argmax per graph
        indicator = torch.zeros_like(probs)
        n_graphs = batch.batch.max().item() + 1
        for g in range(n_graphs):
            graph_mask = batch.batch == g
            graph_probs = probs[graph_mask]
            if graph_probs.sum() > 0:
                local_idx = graph_probs.argmax()
                global_indices = graph_mask.nonzero(as_tuple=True)[0]
                indicator[global_indices[local_idx]] = 1.0
        return indicator.detach()
    else:
        raise ValueError(f"Unknown receiver_mode: {receiver_mode!r}")


def train_shot_epoch(
    model: TwoStageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float = 2.0,
    receiver_mode: str = "oracle",
) -> float:
    """Train shot head for one epoch.

    Args:
        pos_weight: Weight for positive (shot) class in BCE loss.
        receiver_mode: "oracle" | "predicted" | "none".

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    # Keep backbone and receiver_head in eval mode during Stage 2 training
    model.backbone.eval()
    model.receiver_head.eval()

    total_loss = 0.0
    n_batches = 0
    pw = torch.tensor([pos_weight], device=device)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        receiver_indicator = _get_receiver_indicator(
            model, batch, receiver_mode, device,
        )

        # Graph-level features (corner_side)
        n_graphs = batch.batch.max().item() + 1
        graph_features = batch.corner_side.view(n_graphs, 1).float()

        shot_logit = model.predict_shot(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, graph_features, receiver_indicator,
        )

        target = batch.shot_label.float().view(-1, 1).to(device)
        loss = F.binary_cross_entropy_with_logits(
            shot_logit, target, pos_weight=pw,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_receiver(
    model: TwoStageModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate receiver prediction (Stage 1).

    Returns dict with:
        top1_acc: fraction of graphs where argmax == true receiver
        top3_acc: fraction where true receiver is in top-3
        n_labeled: number of graphs with receiver labels
        per_graph: list of per-graph dicts with probs, label, top1, top3
    """
    model.eval()
    per_graph = []

    for batch in loader:
        batch = batch.to(device)

        probs = model.predict_receiver(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.receiver_mask, batch.batch,
        )

        n_graphs = batch.batch.max().item() + 1
        for g in range(n_graphs):
            if not batch.has_receiver_label[g]:
                continue

            graph_mask = batch.batch == g
            g_probs = probs[graph_mask]
            g_label = batch.receiver_label[graph_mask]
            g_recv_mask = batch.receiver_mask[graph_mask]

            # True receiver local index
            true_idx = g_label.argmax().item()

            # Only consider masked (valid candidate) nodes
            candidate_probs = g_probs[g_recv_mask]
            candidate_indices = g_recv_mask.nonzero(as_tuple=True)[0]

            # Rank candidates by probability
            sorted_indices = candidate_probs.argsort(descending=True)
            ranked_candidate_indices = candidate_indices[sorted_indices]

            top1 = ranked_candidate_indices[0].item() == true_idx
            top3 = true_idx in ranked_candidate_indices[:3].tolist()

            per_graph.append({
                "top1": top1,
                "top3": top3,
                "n_candidates": int(g_recv_mask.sum().item()),
            })

    n_labeled = len(per_graph)
    if n_labeled == 0:
        return {"top1_acc": 0.0, "top3_acc": 0.0, "n_labeled": 0, "per_graph": []}

    top1_acc = sum(g["top1"] for g in per_graph) / n_labeled
    top3_acc = sum(g["top3"] for g in per_graph) / n_labeled

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "n_labeled": n_labeled,
        "per_graph": per_graph,
    }


@torch.no_grad()
def eval_shot(
    model: TwoStageModel,
    loader: DataLoader,
    device: torch.device,
    receiver_mode: str = "oracle",
) -> Dict:
    """Evaluate shot prediction (Stage 2).

    Args:
        receiver_mode: "oracle" | "predicted" | "none".

    Returns dict with:
        auc: ROC-AUC (0.5 if single class)
        f1: F1 at optimal threshold
        accuracy: at 0.5 threshold
        probs: list of predicted probabilities
        labels: list of true labels
        n_samples: number of graphs
    """
    from sklearn.metrics import roc_auc_score, f1_score

    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        receiver_indicator = _get_receiver_indicator(
            model, batch, receiver_mode, device,
        )

        n_graphs = batch.batch.max().item() + 1
        graph_features = batch.corner_side.view(n_graphs, 1).float()

        shot_logit = model.predict_shot(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, graph_features, receiver_indicator,
        )

        probs = torch.sigmoid(shot_logit).squeeze(-1).cpu().numpy()
        labels = batch.shot_label.float().cpu().numpy()

        if probs.ndim == 0:
            probs = np.array([probs.item()])
        if labels.ndim == 0:
            labels = np.array([labels.item()])

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    n_samples = len(all_labels)

    # AUC (handle single-class edge case)
    if len(np.unique(all_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    # F1 at optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Accuracy at 0.5
    preds_05 = (all_probs >= 0.5).astype(int)
    accuracy = (preds_05 == all_labels).mean()

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "accuracy": float(accuracy),
        "probs": all_probs.tolist(),
        "labels": all_labels.tolist(),
        "n_samples": n_samples,
        "n_positive": int(all_labels.sum()),
    }


# ---------------------------------------------------------------------------
# Full training pipeline for one fold
# ---------------------------------------------------------------------------


def train_fold(
    model: TwoStageModel,
    train_data: List,
    val_data: List,
    device: torch.device,
    receiver_lr: float = 1e-3,
    receiver_epochs: int = 100,
    receiver_patience: int = 20,
    receiver_weight_decay: float = 1e-3,
    shot_lr: float = 1e-3,
    shot_epochs: int = 100,
    shot_patience: int = 20,
    shot_weight_decay: float = 1e-3,
    shot_pos_weight: float = 2.0,
    batch_size: int = 8,
    receiver_mode: str = "oracle",
) -> TwoStageModel:
    """Train both stages sequentially on one fold.

    Phase 1: Train receiver head with early stopping on val receiver loss.
    Phase 2: Freeze receiver head, train shot head with early stopping on val shot loss.

    Returns the trained model (best checkpoint).
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ---- Phase 1: Receiver ----
    # In pretrained mode: train projection layers + receiver head (backbone frozen)
    # In scratch mode: train entire backbone + receiver head
    trainable_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "receiver_head" in name or "node_proj" in name or "edge_proj" in name:
            trainable_params.append(param)
        elif model.backbone.mode == "scratch" and "backbone" in name:
            trainable_params.append(param)

    if trainable_params:
        optimizer = Adam(trainable_params, lr=receiver_lr, weight_decay=receiver_weight_decay)
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(1, receiver_epochs + 1):
            train_loss = train_receiver_epoch(model, train_loader, optimizer, device)

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_aug = TwoStageModel._augment_with_receiver(batch.x)
                    node_emb = model.backbone(x_aug, batch.edge_index, batch.edge_attr)
                    logits = model.receiver_head(node_emb)
                    loss = receiver_loss(
                        logits, batch.receiver_label, batch.receiver_mask,
                        batch.batch, batch.has_receiver_label,
                    )
                    val_loss += loss.item()
                    n_val += 1

            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= receiver_patience:
                logger.debug("Stage 1 early stop at epoch %d", epoch)
                break

        model.load_state_dict(best_state)

    # ---- Phase 2: Shot ----
    # Freeze receiver head
    for param in model.receiver_head.parameters():
        param.requires_grad = False

    # In pretrained mode: train projection layers + shot head (backbone + receiver frozen)
    # In scratch mode: train entire backbone + shot head (receiver frozen)
    shot_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "shot_head" in name or "node_proj" in name or "edge_proj" in name:
            shot_params.append(param)
        elif model.backbone.mode == "scratch" and "backbone" in name:
            shot_params.append(param)

    if shot_params:
        optimizer = Adam(shot_params, lr=shot_lr, weight_decay=shot_weight_decay)
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(1, shot_epochs + 1):
            train_loss = train_shot_epoch(
                model, train_loader, optimizer, device,
                pos_weight=shot_pos_weight, receiver_mode=receiver_mode,
            )

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            pw = torch.tensor([shot_pos_weight], device=device)
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    recv_ind = _get_receiver_indicator(model, batch, receiver_mode, device)
                    n_g = batch.batch.max().item() + 1
                    gf = batch.corner_side.view(n_g, 1).float()
                    logit = model.predict_shot(
                        batch.x, batch.edge_index, batch.edge_attr,
                        batch.batch, gf, recv_ind,
                    )
                    target = batch.shot_label.float().view(-1, 1).to(device)
                    loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)
                    val_loss += loss.item()
                    n_val += 1

            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= shot_patience:
                logger.debug("Stage 2 early stop at epoch %d", epoch)
                break

        model.load_state_dict(best_state)

    # Unfreeze receiver head for next fold (clean state)
    for param in model.receiver_head.parameters():
        param.requires_grad = True

    return model
