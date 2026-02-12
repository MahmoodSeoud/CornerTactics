#!/usr/bin/env python3
"""
Phase 1: Train USSF Backbone
============================
Train CrystalConv GNN on USSF counterattack data to create pretrained weights
for transfer learning to corner kick prediction.

Trains two backbones:
1. Dense adjacency (hypothesis: transfers better to corners)
2. Normal adjacency (USSF default, control condition)

Architecture matches USSF paper:
- 3 CGConv layers with 128 hidden channels
- Global mean pooling
- Dense head: 128 -> dropout(0.5) -> 128 -> dropout(0.5) -> 1
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# Suppress torch-geometric warnings
import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')
warnings.filterwarnings('ignore', message='.*torch-sparse.*')

# Configuration
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
LOGS_DIR = Path(__file__).parent / "logs"
COMBINED_PKL = DATA_DIR / "combined.pkl"


class CounterattackGNN(nn.Module):
    """
    CrystalConv GNN for counterattack prediction.

    Architecture:
    - 3 CGConv layers with edge features
    - Global mean pooling
    - 2-layer MLP head with dropout
    """

    def __init__(
        self,
        node_features: int = 12,
        edge_features: int = 6,
        hidden_channels: int = 128,
        num_conv_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels

        # First conv layer: node_features -> hidden_channels
        self.conv1 = CGConv(
            channels=node_features,
            dim=edge_features,
            aggr='add'
        )
        # CGConv outputs same dimension as input in PyG, so we need a linear layer
        self.lin_in = nn.Linear(node_features, hidden_channels)

        # Remaining conv layers: hidden_channels -> hidden_channels
        self.convs = nn.ModuleList([
            CGConv(
                channels=hidden_channels,
                dim=edge_features,
                aggr='add'
            )
            for _ in range(num_conv_layers - 1)
        ])

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass."""
        # First conv
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        # Remaining convs
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Head
        x = self.head(x)
        return x

    def get_backbone_state(self) -> Dict:
        """Extract backbone (conv layers) state dict for transfer."""
        return {
            'conv1': self.conv1.state_dict(),
            'lin_in': self.lin_in.state_dict(),
            'convs': [conv.state_dict() for conv in self.convs],
            'config': {
                'node_features': self.node_features,
                'edge_features': self.edge_features,
                'hidden_channels': self.hidden_channels,
                'num_conv_layers': len(self.convs) + 1
            }
        }


def load_ussf_data(adj_type: str = 'normal') -> Tuple[List[Data], np.ndarray]:
    """
    Load USSF data and convert to PyG Data objects.

    Args:
        adj_type: Adjacency type ('normal', 'dense', 'delaunay', etc.)

    Returns:
        List of PyG Data objects and labels array
    """
    print(f"Loading USSF data with '{adj_type}' adjacency...")

    with open(COMBINED_PKL, 'rb') as f:
        data = pickle.load(f)

    x_list = data[adj_type]['x']  # Node features
    a_list = data[adj_type]['a']  # Adjacency matrices
    e_list = data[adj_type]['e']  # Edge features
    labels = np.array(data['binary']).flatten()

    total_graphs = len(x_list)
    print(f"Converting {total_graphs} graphs to PyG format...")

    pyg_data_list = []

    for i, (x, a, e) in enumerate(zip(x_list, a_list, e_list)):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  Processing graph {i+1}/{total_graphs}...")
        # Convert numpy to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Handle scipy sparse matrices
        if sp.issparse(a):
            # Convert scipy sparse to dense, then to torch
            a_dense = a.toarray()
        else:
            a_dense = a

        # Convert dense adjacency to sparse edge_index
        a_tensor = torch.tensor(a_dense, dtype=torch.float32)
        edge_index, _ = dense_to_sparse(a_tensor)

        # Edge features need to match edge_index
        # In USSF data, e has shape (num_edges, 6)
        # edge_index has shape (2, num_edges)
        num_edges = edge_index.shape[1]

        if e.shape[0] == num_edges:
            e_tensor = torch.tensor(e, dtype=torch.float32)
        else:
            # Need to extract edge features matching edge_index
            # Build mapping from (src, dst) to edge feature index
            # USSF stores edges in COO format row-major
            n_nodes = x.shape[0]

            # Create a lookup for edge features
            edge_to_feat = {}
            edge_idx = 0
            for i_src in range(n_nodes):
                for i_dst in range(n_nodes):
                    if a_dense[i_src, i_dst] > 0:
                        edge_to_feat[(i_src, i_dst)] = edge_idx
                        edge_idx += 1

            # Map edge_index to edge features
            edge_features = []
            for j in range(num_edges):
                src, dst = edge_index[0, j].item(), edge_index[1, j].item()
                if (src, dst) in edge_to_feat:
                    feat_idx = edge_to_feat[(src, dst)]
                    if feat_idx < e.shape[0]:
                        edge_features.append(e[feat_idx])
                    else:
                        edge_features.append(np.zeros(e.shape[1]))
                else:
                    edge_features.append(np.zeros(e.shape[1]))

            e_tensor = torch.tensor(np.array(edge_features), dtype=torch.float32)

        y_tensor = torch.tensor([labels[i]], dtype=torch.float32)

        pyg_data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=e_tensor,
            y=y_tensor
        )
        pyg_data_list.append(pyg_data)

    print(f"Loaded {len(pyg_data_list)} graphs")
    return pyg_data_list, labels


def sequence_aware_split(
    data_list: List[Data],
    labels: np.ndarray,
    test_size: float = 0.3,
    seed: int = 15
) -> Tuple[List[Data], List[Data]]:
    """
    Split data maintaining sequence order (as done in USSF paper).

    Uses stratified split with seed=15 for reproducibility.
    """
    indices = np.arange(len(data_list))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]

    print(f"Split: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs

            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            if probs.ndim == 0:
                probs = np.array([probs])
                preds = np.array([preds])

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch.y.squeeze().cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return {
        'loss': total_loss / len(loader.dataset),
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_model(
    adj_type: str,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_channels: int = 128,
    num_conv_layers: int = 3,
    dropout: float = 0.5,
    device: Optional[str] = None
) -> Tuple[CounterattackGNN, Dict]:
    """
    Train the full model.

    Returns trained model and training history.
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load data
    data_list, labels = load_ussf_data(adj_type)

    # Split
    train_data, test_data = sequence_aware_split(data_list, labels)

    # Further split train into train/val
    train_labels = np.array([d.y.item() for d in train_data])
    train_data, val_data = sequence_aware_split(train_data, train_labels, test_size=0.15, seed=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = CounterattackGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=hidden_channels,
        num_conv_layers=num_conv_layers,
        dropout=dropout
    ).to(device)

    print(f"\nModel architecture:\n{model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_val_auc': 0
    }

    best_val_auc = 0
    best_model_state = None

    print(f"\nTraining with {adj_type} adjacency for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step(val_metrics['auc'])

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        # Check best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_val_auc'] = best_val_auc
            marker = '*'
        else:
            marker = ''

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f} {marker}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device)
    history['test_metrics'] = test_metrics

    print("-" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {history['best_epoch']}")
    print(f"Test metrics:")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")

    return model, history


def save_backbone(
    model: CounterattackGNN,
    adj_type: str,
    history: Dict
):
    """Save backbone weights and training info."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save backbone state
    backbone_path = WEIGHTS_DIR / f"ussf_backbone_{adj_type}.pt"
    torch.save(model.get_backbone_state(), backbone_path)
    print(f"Saved backbone to {backbone_path}")

    # Save full model
    full_path = WEIGHTS_DIR / f"ussf_full_model_{adj_type}.pt"
    torch.save(model.state_dict(), full_path)
    print(f"Saved full model to {full_path}")

    # Save history
    history_path = WEIGHTS_DIR / f"ussf_training_history_{adj_type}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved history to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train USSF backbone')
    parser.add_argument('--adj-type', type=str, default='dense',
                        choices=['normal', 'dense', 'delaunay', 'dense_ap', 'dense_dp'],
                        help='Adjacency type to use')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Hidden channel dimension')
    parser.add_argument('--num-conv-layers', type=int, default=3,
                        help='Number of CGConv layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--train-all', action='store_true',
                        help='Train both dense and normal adjacency types')

    args = parser.parse_args()

    # Create log directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"phase1_training_{timestamp}.log"

    print(f"Phase 1: Train USSF Backbone")
    print(f"=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Arguments: {vars(args)}")
    print(f"=" * 60)

    if args.train_all:
        adj_types = ['dense', 'normal']
    else:
        adj_types = [args.adj_type]

    results = {}

    for adj_type in adj_types:
        print(f"\n{'='*60}")
        print(f"Training with {adj_type.upper()} adjacency")
        print(f"{'='*60}")

        model, history = train_model(
            adj_type=adj_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_channels=args.hidden_channels,
            num_conv_layers=args.num_conv_layers,
            dropout=args.dropout,
            device=args.device
        )

        save_backbone(model, adj_type, history)
        results[adj_type] = history

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for adj_type, history in results.items():
        print(f"\n{adj_type.upper()}:")
        print(f"  Best Val AUC: {history['best_val_auc']:.4f} (epoch {history['best_epoch']})")
        print(f"  Test AUC:     {history['test_metrics']['auc']:.4f}")
        print(f"  Test Acc:     {history['test_metrics']['accuracy']:.4f}")

    print(f"\nWeights saved to: {WEIGHTS_DIR}")
    print(f"Done!")


if __name__ == "__main__":
    main()
