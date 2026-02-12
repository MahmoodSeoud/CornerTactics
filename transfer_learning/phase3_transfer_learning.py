#!/usr/bin/env python3
"""
Phase 3: Transfer Learning Experiments
=======================================
Run 6 experimental conditions to test whether USSF-pretrained GNN representations
transfer to corner kick shot prediction.

Experimental Conditions:
    A: USSF pretrained + dense adjacency + frozen backbone
    B: USSF pretrained + normal adjacency + frozen backbone
    C: USSF pretrained + dense adjacency + unfrozen (fine-tuned, lr=1e-5)
    D: Random initialization + dense adjacency
    E: Random initialization + normal adjacency
    F: Majority class baseline

The dataset has 57 corners from 7 matches. We use match-based splitting
to prevent data leakage (all corners from a match stay together).
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')
warnings.filterwarnings('ignore', message='.*torch-sparse.*')

# Configuration
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent / "results"


class TransferGNN(nn.Module):
    """
    Transfer learning GNN for corner kick prediction.

    Can load pretrained USSF backbone with frozen or unfrozen conv layers.
    Uses a smaller head suitable for the small corner kick dataset (57 samples).
    """

    def __init__(
        self,
        node_features: int = 12,
        edge_features: int = 6,
        hidden_channels: int = 128,
        num_conv_layers: int = 3,
        head_hidden: int = 32,
        head_dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.freeze_backbone = freeze_backbone

        # Backbone: CGConv layers (matches USSF architecture)
        self.conv1 = CGConv(
            channels=node_features,
            dim=edge_features,
            aggr='add'
        )
        self.lin_in = nn.Linear(node_features, hidden_channels)

        self.convs = nn.ModuleList([
            CGConv(
                channels=hidden_channels,
                dim=edge_features,
                aggr='add'
            )
            for _ in range(num_conv_layers - 1)
        ])

        # Small head for 57 samples (down from USSF's 128-128-1)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.lin_in.parameters():
            param.requires_grad = True
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True

    def load_pretrained_backbone(self, backbone_state: Dict):
        """Load pretrained weights into backbone layers."""
        # Load conv1
        self.conv1.load_state_dict(backbone_state['conv1'])

        # Load lin_in
        self.lin_in.load_state_dict(backbone_state['lin_in'])

        # Load remaining convs
        for i, conv_state in enumerate(backbone_state['convs']):
            self.convs[i].load_state_dict(conv_state)

        print(f"  Loaded pretrained backbone weights")

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass."""
        # Backbone
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Head
        x = self.head(x)
        return x

    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def prepare_pyg_data(
    corners: List[Dict],
    adj_type: str = 'dense'
) -> List[Data]:
    """
    Convert corner samples to PyG Data objects ready for training.

    Args:
        corners: List of corner samples with 'graphs' and 'labels'
        adj_type: Which adjacency type data was transformed with

    Returns:
        List of PyG Data objects with labels
    """
    data_list = []

    for sample in corners:
        # Get the delivery frame graph (should be single graph per corner)
        graph = sample['graphs'][0]

        # Get binary shot label
        label = float(sample['labels']['shot_binary'])

        # Create PyG Data with label
        pyg_data = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=torch.tensor([label], dtype=torch.float32)
        )

        # Store metadata for debugging
        pyg_data.match_id = sample['match_id']
        pyg_data.corner_time = sample['corner_time']

        data_list.append(pyg_data)

    return data_list


def match_based_split(
    corners: List[Dict],
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split corners into train/val/test keeping matches together.

    Strategy: With 7 matches, use 5 for train, 1 for val, 1 for test.
    Select val/test matches to have reasonable positive rate.

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)

    # Group corners by match
    match_to_indices = {}
    for i, corner in enumerate(corners):
        match_id = corner['match_id']
        if match_id not in match_to_indices:
            match_to_indices[match_id] = []
        match_to_indices[match_id].append(i)

    # Calculate positive rate per match
    match_stats = {}
    for match_id, indices in match_to_indices.items():
        labels = [corners[i]['labels']['shot_binary'] for i in indices]
        pos_rate = sum(labels) / len(labels)
        match_stats[match_id] = {
            'indices': indices,
            'count': len(indices),
            'positives': sum(labels),
            'pos_rate': pos_rate
        }

    print("\nMatch distribution:")
    for match_id, stats in sorted(match_stats.items()):
        print(f"  {match_id}: {stats['count']} corners, "
              f"{stats['positives']} shots ({stats['pos_rate']*100:.1f}%)")

    # Sort matches by size (put medium-sized ones aside for val/test)
    matches_by_size = sorted(match_stats.items(), key=lambda x: x[1]['count'])

    # Select val and test matches (prefer medium-sized with some positives)
    # Shuffle to avoid always picking same matches
    match_ids = list(match_to_indices.keys())
    np.random.shuffle(match_ids)

    # Pick matches for val/test that have at least 1 positive if possible
    matches_with_pos = [m for m in match_ids if match_stats[m]['positives'] > 0]
    matches_without_pos = [m for m in match_ids if match_stats[m]['positives'] == 0]

    # Prioritize matches with positives for val/test
    if len(matches_with_pos) >= 2:
        val_match = matches_with_pos[0]
        test_match = matches_with_pos[1]
    elif len(matches_with_pos) == 1:
        val_match = matches_with_pos[0]
        test_match = matches_without_pos[0] if matches_without_pos else match_ids[-1]
    else:
        val_match = match_ids[0]
        test_match = match_ids[1]

    train_matches = [m for m in match_ids if m not in [val_match, test_match]]

    # Collect indices
    train_indices = []
    for m in train_matches:
        train_indices.extend(match_to_indices[m])
    val_indices = match_to_indices[val_match]
    test_indices = match_to_indices[test_match]

    print(f"\nSplit:")
    print(f"  Train: {len(train_indices)} corners from {len(train_matches)} matches")
    print(f"  Val:   {len(val_indices)} corners from 1 match ({val_match})")
    print(f"  Test:  {len(test_indices)} corners from 1 match ({test_match})")

    return train_indices, val_indices, test_indices


def compute_class_weights(labels: List[float]) -> torch.Tensor:
    """Compute class weights for imbalanced binary classification."""
    labels = np.array(labels)
    n_samples = len(labels)
    n_pos = labels.sum()
    n_neg = n_samples - n_pos

    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])

    # Inverse frequency weighting
    w_neg = n_samples / (2 * n_neg)
    w_pos = n_samples / (2 * n_pos)

    return torch.tensor([w_neg, w_pos])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.squeeze(),
                pos_weight=pos_weight.to(device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs])

            preds = (probs > 0.5).astype(int)

            labels = batch.y.squeeze().cpu().numpy()
            if labels.ndim == 0:
                labels = np.array([labels])

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Handle edge cases for AUC
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        auc = 0.5  # Cannot compute AUC with single class
    else:
        auc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return {
        'loss': total_loss / n_samples if n_samples > 0 else 0,
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': n_samples,
        'n_positive': int(all_labels.sum()),
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }


def run_experiment(
    condition: str,
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    adj_type: str,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 8,
    patience: int = 10,
    device: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """
    Run a single experimental condition.

    Returns:
        Dictionary with training history and final metrics
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Condition {condition}: adj={adj_type}, pretrained={pretrained}, frozen={freeze_backbone}")
    print(f"{'='*60}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Compute class weights from training labels
    train_labels = [d.y.item() for d in train_data]
    class_weights = compute_class_weights(train_labels)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]])

    print(f"Training samples: {len(train_data)} ({sum(train_labels)} positive)")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Class weights: neg={class_weights[0]:.3f}, pos={class_weights[1]:.3f}")

    # Create model
    model = TransferGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=128,
        num_conv_layers=3,
        head_hidden=32,
        head_dropout=0.3,
        freeze_backbone=freeze_backbone
    ).to(device)

    # Load pretrained weights if applicable
    if pretrained:
        backbone_path = WEIGHTS_DIR / f"ussf_backbone_{adj_type}.pt"
        if backbone_path.exists():
            backbone_state = torch.load(backbone_path, map_location=device)
            model.load_pretrained_backbone(backbone_state)
        else:
            print(f"  WARNING: Pretrained weights not found at {backbone_path}")
            print(f"  Using random initialization instead")

    trainable = model.count_trainable_params()
    total = model.count_total_params()
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop with early stopping
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_val_auc': 0.0
    }

    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        # Check for improvement
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch
            history['best_val_auc'] = best_val_auc
            epochs_without_improvement = 0
            marker = '*'
        else:
            epochs_without_improvement += 1
            marker = ''

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:2d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.3f} {marker}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device)
    train_metrics = evaluate(model, train_loader, device)

    print(f"\nResults for Condition {condition}:")
    print(f"  Best val AUC: {history['best_val_auc']:.4f} (epoch {history['best_epoch']})")
    print(f"  Test AUC:     {test_metrics['auc']:.4f}")
    print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")
    print(f"  Test Prec:    {test_metrics['precision']:.4f}")
    print(f"  Test Recall:  {test_metrics['recall']:.4f}")

    return {
        'condition': condition,
        'config': {
            'adj_type': adj_type,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience
        },
        'history': history,
        'train_metrics': train_metrics,
        'val_metrics': evaluate(model, val_loader, device),
        'test_metrics': test_metrics
    }


def run_majority_baseline(
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data]
) -> Dict:
    """
    Condition F: Majority class baseline.

    Predict the majority class for all samples.
    """
    print(f"\n{'='*60}")
    print("Condition F: Majority Class Baseline")
    print(f"{'='*60}")

    # Get training labels
    train_labels = np.array([d.y.item() for d in train_data])
    majority_class = 1.0 if train_labels.mean() > 0.5 else 0.0

    print(f"Training positive rate: {train_labels.mean():.3f}")
    print(f"Majority class: {int(majority_class)}")

    # Evaluate on each split
    results = {}
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        labels = np.array([d.y.item() for d in data])
        preds = np.full_like(labels, majority_class)
        probs = np.full_like(labels, train_labels.mean())  # Use training rate as prob

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(labels, probs)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )

        results[f'{split_name}_metrics'] = {
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(labels),
            'n_positive': int(labels.sum())
        }

    print(f"\nResults for Condition F (Majority Baseline):")
    print(f"  Test AUC:     {results['test_metrics']['auc']:.4f}")
    print(f"  Test Acc:     {results['test_metrics']['accuracy']:.4f}")

    return {
        'condition': 'F',
        'config': {
            'majority_class': int(majority_class),
            'training_positive_rate': float(train_labels.mean())
        },
        'train_metrics': results['train_metrics'],
        'val_metrics': results['val_metrics'],
        'test_metrics': results['test_metrics']
    }


def bootstrap_ci(
    metrics: Dict,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for metrics.

    Returns dict mapping metric name to (lower, upper) CI bounds.
    """
    np.random.seed(seed)

    if 'labels' not in metrics or 'probabilities' not in metrics:
        return {}

    labels = np.array(metrics['labels'])
    probs = np.array(metrics['probabilities'])
    n = len(labels)

    if n < 5:
        return {}

    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_probs = probs[idx]

        if len(np.unique(boot_labels)) < 2:
            continue

        aucs.append(roc_auc_score(boot_labels, boot_probs))

    if not aucs:
        return {}

    alpha = 1 - ci_level
    lower = np.percentile(aucs, alpha / 2 * 100)
    upper = np.percentile(aucs, (1 - alpha / 2) * 100)

    return {
        'auc_ci': (lower, upper)
    }


def run_multi_seed_experiments(
    corners_dense: List[Dict],
    corners_normal: List[Dict],
    conditions: List[str],
    seeds: List[int],
    epochs: int = 50,
    batch_size: int = 8,
    patience: int = 10,
    device: Optional[str] = None
) -> Dict:
    """
    Run all conditions across multiple seeds and aggregate results.

    Returns:
        Dict with aggregated results per condition
    """
    all_seed_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        # Create split for this seed
        train_idx, val_idx, test_idx = match_based_split(corners_dense, seed=seed)

        # Prepare data
        data_dense = prepare_pyg_data(corners_dense, adj_type='dense')
        data_normal = prepare_pyg_data(corners_normal, adj_type='normal')

        train_dense = [data_dense[i] for i in train_idx]
        val_dense = [data_dense[i] for i in val_idx]
        test_dense = [data_dense[i] for i in test_idx]

        train_normal = [data_normal[i] for i in train_idx]
        val_normal = [data_normal[i] for i in val_idx]
        test_normal = [data_normal[i] for i in test_idx]

        condition_configs = {
            'A': {'adj_type': 'dense', 'pretrained': True, 'freeze_backbone': True, 'lr': 1e-4},
            'B': {'adj_type': 'normal', 'pretrained': True, 'freeze_backbone': True, 'lr': 1e-4},
            'C': {'adj_type': 'dense', 'pretrained': True, 'freeze_backbone': False, 'lr': 1e-5},
            'D': {'adj_type': 'dense', 'pretrained': False, 'freeze_backbone': False, 'lr': 1e-4},
            'E': {'adj_type': 'normal', 'pretrained': False, 'freeze_backbone': False, 'lr': 1e-4},
        }

        for condition in conditions:
            if condition == 'F':
                result = run_majority_baseline(train_dense, val_dense, test_dense)
            elif condition in condition_configs:
                config = condition_configs[condition]

                if config['adj_type'] == 'dense':
                    train_data, val_data, test_data = train_dense, val_dense, test_dense
                else:
                    train_data, val_data, test_data = train_normal, val_normal, test_normal

                result = run_experiment(
                    condition=condition,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    adj_type=config['adj_type'],
                    pretrained=config['pretrained'],
                    freeze_backbone=config['freeze_backbone'],
                    lr=config['lr'],
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    device=device,
                    seed=seed
                )
            else:
                continue

            all_seed_results[seed][condition] = result

    # Aggregate results across seeds
    aggregated = {}
    for condition in conditions:
        aucs = []
        accs = []
        for seed in seeds:
            if condition in all_seed_results[seed]:
                aucs.append(all_seed_results[seed][condition]['test_metrics']['auc'])
                accs.append(all_seed_results[seed][condition]['test_metrics']['accuracy'])

        if aucs:
            aggregated[condition] = {
                'test_auc_mean': np.mean(aucs),
                'test_auc_std': np.std(aucs),
                'test_acc_mean': np.mean(accs),
                'test_acc_std': np.std(accs),
                'n_seeds': len(aucs),
                'aucs': aucs,
                'accs': accs
            }

    return {
        'per_seed': all_seed_results,
        'aggregated': aggregated
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Transfer Learning Experiments')
    parser.add_argument('--conditions', type=str, nargs='+', default=['A', 'B', 'C', 'D', 'E', 'F'],
                        help='Which conditions to run (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per condition')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (single seed mode)')
    parser.add_argument('--multi-seed', action='store_true',
                        help='Run with multiple seeds for robust estimates')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1234],
                        help='Seeds to use in multi-seed mode')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Compute bootstrap confidence intervals')

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3: Transfer Learning Experiments")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Conditions: {args.conditions}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load both dense and normal transformed datasets
    dense_path = DATA_DIR / "dfl_corners_ussf_format_dense.pkl"
    normal_path = DATA_DIR / "dfl_corners_ussf_format_normal.pkl"

    print(f"\n--- Loading Data ---")
    with open(dense_path, 'rb') as f:
        corners_dense = pickle.load(f)
    print(f"Loaded {len(corners_dense)} corners (dense adjacency)")

    with open(normal_path, 'rb') as f:
        corners_normal = pickle.load(f)
    print(f"Loaded {len(corners_normal)} corners (normal adjacency)")

    # Multi-seed mode
    if args.multi_seed:
        print(f"\nRunning multi-seed experiment with seeds: {args.seeds}")

        multi_results = run_multi_seed_experiments(
            corners_dense=corners_dense,
            corners_normal=corners_normal,
            conditions=args.conditions,
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device
        )

        # Print aggregated results
        print(f"\n{'='*80}")
        print("AGGREGATED RESULTS (across {} seeds)".format(len(args.seeds)))
        print(f"{'='*80}")
        print(f"\n{'Condition':<12} {'Test AUC (mean±std)':<25} {'Test Acc (mean±std)':<25}")
        print("-" * 70)

        for cond, agg in sorted(multi_results['aggregated'].items()):
            auc_str = f"{agg['test_auc_mean']:.4f} ± {agg['test_auc_std']:.4f}"
            acc_str = f"{agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f}"
            print(f"{cond:<12} {auc_str:<25} {acc_str:<25}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = RESULTS_DIR / f"phase3_multiseed_{timestamp}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(multi_results, f)
        print(f"\nResults saved to: {results_path}")

        # Save summary
        import json
        summary = {
            'timestamp': timestamp,
            'seeds': args.seeds,
            'aggregated': {
                cond: {
                    'test_auc_mean': agg['test_auc_mean'],
                    'test_auc_std': agg['test_auc_std'],
                    'test_acc_mean': agg['test_acc_mean'],
                    'test_acc_std': agg['test_acc_std'],
                    'n_seeds': agg['n_seeds']
                }
                for cond, agg in multi_results['aggregated'].items()
            }
        }
        summary_path = RESULTS_DIR / f"phase3_multiseed_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        print(f"\nPhase 3 Complete!")
        return

    # Single seed mode
    print(f"Seed: {args.seed}")

    # Create match-based split (same for both adjacency types)
    train_idx, val_idx, test_idx = match_based_split(corners_dense, seed=args.seed)

    # Prepare PyG data
    data_dense = prepare_pyg_data(corners_dense, adj_type='dense')
    data_normal = prepare_pyg_data(corners_normal, adj_type='normal')

    train_dense = [data_dense[i] for i in train_idx]
    val_dense = [data_dense[i] for i in val_idx]
    test_dense = [data_dense[i] for i in test_idx]

    train_normal = [data_normal[i] for i in train_idx]
    val_normal = [data_normal[i] for i in val_idx]
    test_normal = [data_normal[i] for i in test_idx]

    # Condition configurations
    condition_configs = {
        'A': {'adj_type': 'dense', 'pretrained': True, 'freeze_backbone': True, 'lr': 1e-4},
        'B': {'adj_type': 'normal', 'pretrained': True, 'freeze_backbone': True, 'lr': 1e-4},
        'C': {'adj_type': 'dense', 'pretrained': True, 'freeze_backbone': False, 'lr': 1e-5},
        'D': {'adj_type': 'dense', 'pretrained': False, 'freeze_backbone': False, 'lr': 1e-4},
        'E': {'adj_type': 'normal', 'pretrained': False, 'freeze_backbone': False, 'lr': 1e-4},
    }

    # Run experiments
    all_results = {}

    for condition in args.conditions:
        if condition == 'F':
            result = run_majority_baseline(train_dense, val_dense, test_dense)
        elif condition in condition_configs:
            config = condition_configs[condition]

            if config['adj_type'] == 'dense':
                train_data, val_data, test_data = train_dense, val_dense, test_dense
            else:
                train_data, val_data, test_data = train_normal, val_normal, test_normal

            result = run_experiment(
                condition=condition,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                adj_type=config['adj_type'],
                pretrained=config['pretrained'],
                freeze_backbone=config['freeze_backbone'],
                lr=config['lr'],
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                device=args.device,
                seed=args.seed
            )

            # Compute bootstrap CI if requested
            if args.bootstrap:
                ci = bootstrap_ci(result['test_metrics'])
                if ci:
                    result['test_ci'] = ci
        else:
            print(f"Unknown condition: {condition}, skipping")
            continue

        all_results[condition] = result

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Condition':<12} {'Adj Type':<10} {'Pretrained':<12} {'Frozen':<8} {'Test AUC':<10} {'Test Acc':<10}")
    print("-" * 70)

    for cond, result in sorted(all_results.items()):
        if cond == 'F':
            adj = '-'
            pre = '-'
            frz = '-'
        else:
            adj = result['config']['adj_type']
            pre = 'Yes' if result['config']['pretrained'] else 'No'
            frz = 'Yes' if result['config']['freeze_backbone'] else 'No'

        auc = result['test_metrics']['auc']
        acc = result['test_metrics']['accuracy']

        # Add CI if available
        if 'test_ci' in result:
            ci_low, ci_high = result['test_ci']['auc_ci']
            auc_str = f"{auc:.4f} [{ci_low:.3f}-{ci_high:.3f}]"
        else:
            auc_str = f"{auc:.4f}"

        print(f"{cond:<12} {adj:<10} {pre:<12} {frz:<8} {auc_str:<20} {acc:.4f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f"phase3_results_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {results_path}")

    # Also save a summary JSON for easy reading
    summary = {
        'timestamp': timestamp,
        'seed': args.seed,
        'conditions': {}
    }
    for cond, result in all_results.items():
        summary['conditions'][cond] = {
            'test_auc': result['test_metrics']['auc'],
            'test_accuracy': result['test_metrics']['accuracy'],
            'test_n_samples': result['test_metrics']['n_samples'],
            'test_n_positive': result['test_metrics']['n_positive']
        }
        if cond != 'F':
            summary['conditions'][cond].update({
                'adj_type': result['config']['adj_type'],
                'pretrained': result['config']['pretrained'],
                'frozen': result['config']['freeze_backbone']
            })
            if 'history' in result:
                summary['conditions'][cond]['best_epoch'] = result['history']['best_epoch']
                summary['conditions'][cond]['best_val_auc'] = result['history']['best_val_auc']

    import json
    summary_path = RESULTS_DIR / f"phase3_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print(f"\nPhase 3 Complete!")


if __name__ == "__main__":
    main()
