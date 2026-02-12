#!/usr/bin/env python3
"""
Phase 4: Velocity Ablation - Feature Importance via Permutation Testing
=========================================================================

Determines whether velocity features are critical for corner kick predictions
by using permutation importance testing following USSF's ShuffledCounterDataset
methodology.

Ablation Experiments:
    1. Baseline: Evaluate best model on unmodified test set
    2. Velocity ablation: Shuffle velocity features (vx, vy, velocity_mag, velocity_angle)
    3. Position ablation: Shuffle position features (x, y)
    4. Derived position ablation: Shuffle dist/angle features

Interpretation:
    - If AUC drops significantly after velocity shuffle: velocity is critical
    - If AUC drops significantly after position shuffle: position is critical
    - If neither drops: transfer may have failed (model learned nothing)
    - If both drop equally: both feature groups matter

This connects to the 7.5 ECTS finding that position-only achieved AUC=0.50.
If velocity shuffling tanks performance but position shuffling doesn't, we've
experimentally isolated velocity as the critical ingredient.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')
warnings.filterwarnings('ignore', message='.*torch-sparse.*')

# Configuration
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent / "results"

# Feature indices in USSF schema (12 node features)
FEATURE_INDICES = {
    'x': 0,
    'y': 1,
    'vx': 2,
    'vy': 3,
    'velocity_mag': 4,
    'velocity_angle': 5,
    'dist_goal': 6,
    'angle_goal': 7,
    'dist_ball': 8,
    'angle_ball': 9,
    'attacking_team_flag': 10,
    'potential_receiver': 11
}

# Feature groups for ablation
ABLATION_GROUPS = {
    'velocity_raw': [2, 3],                    # vx, vy only
    'velocity_all': [2, 3, 4, 5],              # vx, vy, velocity_mag, velocity_angle
    'position_raw': [0, 1],                    # x, y only
    'position_derived': [6, 7, 8, 9],          # dist_goal, angle_goal, dist_ball, angle_ball
    'position_all': [0, 1, 6, 7, 8, 9],        # all position-related
}


class TransferGNN(nn.Module):
    """
    Transfer learning GNN for corner kick prediction.
    Same architecture as Phase 3.
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

        # Small head for 57 samples
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def load_pretrained_backbone(self, backbone_state: Dict):
        self.conv1.load_state_dict(backbone_state['conv1'])
        self.lin_in.load_state_dict(backbone_state['lin_in'])
        for i, conv_state in enumerate(backbone_state['convs']):
            self.convs[i].load_state_dict(conv_state)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.head(x)
        return x


def prepare_pyg_data(corners: List[Dict]) -> List[Data]:
    """Convert corner samples to PyG Data objects."""
    data_list = []

    for sample in corners:
        graph = sample['graphs'][0]
        label = float(sample['labels']['shot_binary'])

        pyg_data = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=torch.tensor([label], dtype=torch.float32)
        )
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
    Same logic as Phase 3 for reproducibility.
    """
    np.random.seed(seed)

    match_to_indices = {}
    for i, corner in enumerate(corners):
        match_id = corner['match_id']
        if match_id not in match_to_indices:
            match_to_indices[match_id] = []
        match_to_indices[match_id].append(i)

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

    match_ids = list(match_to_indices.keys())
    np.random.shuffle(match_ids)

    matches_with_pos = [m for m in match_ids if match_stats[m]['positives'] > 0]
    matches_without_pos = [m for m in match_ids if match_stats[m]['positives'] == 0]

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

    train_indices = []
    for m in train_matches:
        train_indices.extend(match_to_indices[m])
    val_indices = match_to_indices[val_match]
    test_indices = match_to_indices[test_match]

    return train_indices, val_indices, test_indices


def shuffle_features(
    data_list: List[Data],
    feature_indices: List[int],
    seed: int = 42
) -> List[Data]:
    """
    Shuffle specific node features across all nodes in the dataset.

    This breaks the correlation between these features and the labels
    while preserving the marginal distribution of the features.

    Args:
        data_list: List of PyG Data objects
        feature_indices: Which feature dimensions to shuffle (0-indexed)
        seed: Random seed for reproducibility

    Returns:
        New list of Data objects with shuffled features
    """
    np.random.seed(seed)

    # Collect all node features across graphs
    all_nodes = []
    node_counts = []

    for data in data_list:
        all_nodes.append(data.x.numpy())
        node_counts.append(data.x.shape[0])

    all_nodes = np.concatenate(all_nodes, axis=0)  # (total_nodes, 12)
    total_nodes = all_nodes.shape[0]

    # Create shuffled version
    shuffled_nodes = all_nodes.copy()

    # Shuffle each specified feature dimension independently
    for feat_idx in feature_indices:
        perm = np.random.permutation(total_nodes)
        shuffled_nodes[:, feat_idx] = all_nodes[perm, feat_idx]

    # Rebuild data list with shuffled features
    shuffled_data_list = []
    offset = 0

    for i, data in enumerate(data_list):
        n_nodes = node_counts[i]
        shuffled_x = torch.tensor(shuffled_nodes[offset:offset+n_nodes], dtype=torch.float32)
        offset += n_nodes

        new_data = Data(
            x=shuffled_x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone(),
            y=data.y.clone()
        )
        new_data.match_id = data.match_id
        new_data.corner_time = data.corner_time
        shuffled_data_list.append(new_data)

    return shuffled_data_list


def evaluate(
    model: nn.Module,
    data_list: List[Data],
    device: torch.device,
    batch_size: int = 8
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

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

    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)

    return {
        'auc': auc,
        'accuracy': acc,
        'n_samples': len(all_labels),
        'n_positive': int(all_labels.sum()),
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }


def load_trained_model(
    condition: str,
    adj_type: str,
    results_path: Path,
    weights_dir: Path,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """
    Load a trained model from Phase 3 results.

    For conditions A/B/C (pretrained), we load the backbone weights and
    need to also train the head. Since we don't save the fine-tuned head
    separately, we need to re-train.

    For simplicity, this function loads backbone weights and returns
    an untrained model. The caller should re-train or use saved full weights.
    """
    # Create model with appropriate config
    freeze = condition in ['A', 'B']

    model = TransferGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=128,
        num_conv_layers=3,
        head_hidden=32,
        head_dropout=0.3,
        freeze_backbone=freeze
    ).to(device)

    # Load pretrained backbone
    backbone_path = weights_dir / f"ussf_backbone_{adj_type}.pt"
    if backbone_path.exists():
        backbone_state = torch.load(backbone_path, map_location=device)
        model.load_pretrained_backbone(backbone_state)

    return model, {}


def train_model_for_ablation(
    model: nn.Module,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    patience: int = 10,
    seed: int = 42
) -> nn.Module:
    """
    Train model for ablation evaluation.
    Returns the trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Compute class weights
    train_labels = [d.y.item() for d in train_data]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.squeeze(),
                pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

        # Validate
        val_metrics = evaluate(model, val_data, device, batch_size)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_ablation_experiment(
    condition: str,
    corners: List[Dict],
    weights_dir: Path,
    device: torch.device,
    n_permutations: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run complete ablation experiment for a given condition.

    Trains the model, then evaluates with different feature shufflings.
    Repeats with multiple random seeds to get robust estimates.

    Args:
        condition: 'A', 'B', 'C', etc.
        corners: List of corner samples
        weights_dir: Path to pretrained weights
        device: torch device
        n_permutations: Number of random shuffles per ablation
        seed: Base random seed
        verbose: Print progress

    Returns:
        Dict with baseline and ablation results
    """
    # Condition configs (same as Phase 3)
    condition_configs = {
        'A': {'adj_type': 'dense', 'pretrained': True, 'freeze': True, 'lr': 1e-4},
        'B': {'adj_type': 'normal', 'pretrained': True, 'freeze': True, 'lr': 1e-4},
        'C': {'adj_type': 'dense', 'pretrained': True, 'freeze': False, 'lr': 1e-5},
    }

    if condition not in condition_configs:
        raise ValueError(f"Unknown condition: {condition}")

    config = condition_configs[condition]
    adj_type = config['adj_type']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Ablation for Condition {condition}")
        print(f"  Adjacency: {adj_type}, Frozen: {config['freeze']}")
        print(f"{'='*60}")

    # Split data
    train_idx, val_idx, test_idx = match_based_split(corners, seed=seed)
    data_list = prepare_pyg_data(corners)

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    if verbose:
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create and train model
    model = TransferGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=128,
        num_conv_layers=3,
        head_hidden=32,
        head_dropout=0.3,
        freeze_backbone=config['freeze']
    ).to(device)

    # Load pretrained backbone
    backbone_path = weights_dir / f"ussf_backbone_{adj_type}.pt"
    if backbone_path.exists():
        backbone_state = torch.load(backbone_path, map_location=device)
        model.load_pretrained_backbone(backbone_state)
        if verbose:
            print(f"  Loaded pretrained backbone from {backbone_path}")

    # Train the head
    if verbose:
        print(f"  Training model...")

    model = train_model_for_ablation(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        epochs=50,
        batch_size=8,
        lr=config['lr'],
        patience=10,
        seed=seed
    )

    # Baseline evaluation
    baseline_metrics = evaluate(model, test_data, device)
    if verbose:
        print(f"  Baseline test AUC: {baseline_metrics['auc']:.4f}")

    results = {
        'condition': condition,
        'config': config,
        'seed': seed,
        'n_permutations': n_permutations,
        'baseline': baseline_metrics,
        'ablations': {}
    }

    # Run ablations
    for ablation_name, feature_indices in ABLATION_GROUPS.items():
        if verbose:
            print(f"\n  Ablation: {ablation_name} (features {feature_indices})")

        ablation_aucs = []
        ablation_accs = []

        for perm_i in range(n_permutations):
            perm_seed = seed + perm_i * 1000

            # Shuffle features
            shuffled_test = shuffle_features(test_data, feature_indices, seed=perm_seed)

            # Evaluate
            metrics = evaluate(model, shuffled_test, device)
            ablation_aucs.append(metrics['auc'])
            ablation_accs.append(metrics['accuracy'])

        ablation_result = {
            'features': feature_indices,
            'feature_names': [
                name for name, idx in FEATURE_INDICES.items()
                if idx in feature_indices
            ],
            'aucs': ablation_aucs,
            'accs': ablation_accs,
            'auc_mean': np.mean(ablation_aucs),
            'auc_std': np.std(ablation_aucs),
            'acc_mean': np.mean(ablation_accs),
            'acc_std': np.std(ablation_accs),
            'auc_drop': baseline_metrics['auc'] - np.mean(ablation_aucs),
            'auc_drop_pct': (baseline_metrics['auc'] - np.mean(ablation_aucs)) / max(baseline_metrics['auc'], 0.01) * 100
        }

        results['ablations'][ablation_name] = ablation_result

        if verbose:
            print(f"    AUC: {ablation_result['auc_mean']:.4f} ± {ablation_result['auc_std']:.4f}")
            print(f"    AUC drop: {ablation_result['auc_drop']:.4f} ({ablation_result['auc_drop_pct']:.1f}%)")

    return results


def run_multi_seed_ablation(
    corners: List[Dict],
    weights_dir: Path,
    device: torch.device,
    conditions: List[str] = ['A', 'C'],
    seeds: List[int] = [42, 123, 456, 789, 1234],
    n_permutations: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Run ablation experiments across multiple seeds for robust estimates.
    """
    all_results = {
        'conditions': {},
        'seeds': seeds,
        'n_permutations': n_permutations
    }

    for condition in conditions:
        condition_results = []

        for seed in seeds:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Condition {condition}, Seed {seed}")
                print(f"{'#'*60}")

            result = run_ablation_experiment(
                condition=condition,
                corners=corners,
                weights_dir=weights_dir,
                device=device,
                n_permutations=n_permutations,
                seed=seed,
                verbose=verbose
            )
            condition_results.append(result)

        # Aggregate across seeds
        baseline_aucs = [r['baseline']['auc'] for r in condition_results]

        aggregated = {
            'baseline_auc_mean': np.mean(baseline_aucs),
            'baseline_auc_std': np.std(baseline_aucs),
            'ablations': {}
        }

        # Aggregate each ablation type
        ablation_names = list(condition_results[0]['ablations'].keys())
        for abl_name in ablation_names:
            all_auc_drops = []
            all_auc_means = []

            for r in condition_results:
                all_auc_drops.append(r['ablations'][abl_name]['auc_drop'])
                all_auc_means.append(r['ablations'][abl_name]['auc_mean'])

            aggregated['ablations'][abl_name] = {
                'auc_drop_mean': np.mean(all_auc_drops),
                'auc_drop_std': np.std(all_auc_drops),
                'ablated_auc_mean': np.mean(all_auc_means),
                'ablated_auc_std': np.std(all_auc_means),
                'features': condition_results[0]['ablations'][abl_name]['feature_names']
            }

        all_results['conditions'][condition] = {
            'per_seed': condition_results,
            'aggregated': aggregated
        }

    return all_results


def print_summary(results: Dict) -> None:
    """Print formatted summary of ablation results."""
    print("\n" + "="*80)
    print("PHASE 4: VELOCITY ABLATION RESULTS SUMMARY")
    print("="*80)

    for condition, cond_results in results['conditions'].items():
        agg = cond_results['aggregated']

        print(f"\n{'─'*60}")
        print(f"Condition {condition}")
        print(f"{'─'*60}")
        print(f"Baseline AUC: {agg['baseline_auc_mean']:.4f} ± {agg['baseline_auc_std']:.4f}")
        print()
        print(f"{'Ablation':<20} {'Shuffled Features':<30} {'AUC Drop (mean±std)':<20}")
        print("-"*70)

        # Sort by AUC drop (largest drop first)
        sorted_ablations = sorted(
            agg['ablations'].items(),
            key=lambda x: x[1]['auc_drop_mean'],
            reverse=True
        )

        for abl_name, abl_data in sorted_ablations:
            features = ', '.join(abl_data['features'])
            drop_str = f"{abl_data['auc_drop_mean']:.4f} ± {abl_data['auc_drop_std']:.4f}"
            print(f"{abl_name:<20} {features:<30} {drop_str:<20}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    for condition, cond_results in results['conditions'].items():
        agg = cond_results['aggregated']
        baseline = agg['baseline_auc_mean']

        vel_drop = agg['ablations'].get('velocity_all', {}).get('auc_drop_mean', 0)
        pos_drop = agg['ablations'].get('position_raw', {}).get('auc_drop_mean', 0)

        print(f"\nCondition {condition}:")
        print(f"  Baseline AUC: {baseline:.4f}")
        print(f"  Velocity ablation (vx,vy,mag,angle) drops AUC by: {vel_drop:.4f}")
        print(f"  Position ablation (x,y) drops AUC by: {pos_drop:.4f}")

        # Determine which is more important
        if vel_drop > pos_drop + 0.05:
            print(f"  → Velocity features are MORE important than raw position")
        elif pos_drop > vel_drop + 0.05:
            print(f"  → Position features are MORE important than velocity")
        else:
            print(f"  → Both feature groups have similar importance")

        # Check if either matters
        if vel_drop < 0.02 and pos_drop < 0.02:
            print(f"  ⚠ Neither ablation significantly affects performance - model may not be using these features effectively")

        # Connection to 7.5 ECTS
        if vel_drop > 0.05:
            print(f"  ✓ This supports the hypothesis that velocity is critical for corner prediction")
            print(f"    (Position-only achieved AUC=0.50 in 7.5 ECTS work)")


def main():
    parser = argparse.ArgumentParser(description='Phase 4: Velocity Ablation')
    parser.add_argument('--conditions', type=str, nargs='+', default=['A', 'C'],
                        help='Which conditions to ablate (default: A, C - best performers)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1234],
                        help='Random seeds for robustness')
    parser.add_argument('--n-permutations', type=int, default=10,
                        help='Number of random shuffles per ablation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--single-seed', type=int, default=None,
                        help='Run with single seed only (for debugging)')
    parser.add_argument('--adj-type', type=str, default='dense',
                        help='Adjacency type for data loading')

    args = parser.parse_args()

    print("="*60)
    print("Phase 4: Velocity Ablation")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Conditions: {args.conditions}")
    print(f"Seeds: {args.seeds if args.single_seed is None else [args.single_seed]}")
    print(f"Permutations per ablation: {args.n_permutations}")

    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = DATA_DIR / f"dfl_corners_ussf_format_{args.adj_type}.pkl"
    print(f"\nLoading data from {data_path}")

    with open(data_path, 'rb') as f:
        corners = pickle.load(f)
    print(f"Loaded {len(corners)} corners")

    # Run ablation
    if args.single_seed is not None:
        # Single seed mode (for debugging)
        all_results = {'conditions': {}, 'seeds': [args.single_seed], 'n_permutations': args.n_permutations}

        for condition in args.conditions:
            result = run_ablation_experiment(
                condition=condition,
                corners=corners,
                weights_dir=WEIGHTS_DIR,
                device=device,
                n_permutations=args.n_permutations,
                seed=args.single_seed,
                verbose=True
            )
            all_results['conditions'][condition] = {
                'per_seed': [result],
                'aggregated': {
                    'baseline_auc_mean': result['baseline']['auc'],
                    'baseline_auc_std': 0.0,
                    'ablations': {
                        name: {
                            'auc_drop_mean': data['auc_drop'],
                            'auc_drop_std': 0.0,
                            'ablated_auc_mean': data['auc_mean'],
                            'ablated_auc_std': data['auc_std'],
                            'features': data['feature_names']
                        }
                        for name, data in result['ablations'].items()
                    }
                }
            }
    else:
        # Multi-seed mode
        all_results = run_multi_seed_ablation(
            corners=corners,
            weights_dir=WEIGHTS_DIR,
            device=device,
            conditions=args.conditions,
            seeds=args.seeds,
            n_permutations=args.n_permutations,
            verbose=True
        )

    # Print summary
    print_summary(all_results)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f"phase4_ablation_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {results_path}")

    # Save summary JSON
    import json
    summary = {
        'timestamp': timestamp,
        'conditions': args.conditions,
        'seeds': args.seeds if args.single_seed is None else [args.single_seed],
        'n_permutations': args.n_permutations,
        'results': {}
    }

    for cond, data in all_results['conditions'].items():
        agg = data['aggregated']
        summary['results'][cond] = {
            'baseline_auc': agg['baseline_auc_mean'],
            'baseline_auc_std': agg['baseline_auc_std'],
            'ablations': {
                name: {
                    'auc_drop': abl['auc_drop_mean'],
                    'auc_drop_std': abl['auc_drop_std'],
                    'features': abl['features']
                }
                for name, abl in agg['ablations'].items()
            }
        }

    summary_path = RESULTS_DIR / f"phase4_ablation_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\nPhase 4 Complete!")


if __name__ == "__main__":
    main()
