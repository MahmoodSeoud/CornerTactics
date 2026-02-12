#!/usr/bin/env python3
"""
Phase 5b: Validate USSF Transfer on DFL Open-Play
==================================================

Test whether USSF-pretrained representations generalize to DFL open-play data.

This is a clean evaluation:
- USSF model was trained on USSF counterattack data (20,863 graphs)
- We test it on DFL open-play data (11,967 graphs) it has NEVER seen
- Task: predict shot within 5 seconds

If AUC > 0.55: USSF representations transfer cross-dataset for shot prediction
If AUC ≈ 0.50: USSF representations don't transfer, corner result is likely noise
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent / "results"


class FrozenTransferModel(nn.Module):
    """USSF pretrained backbone (frozen) + linear probe."""

    def __init__(self, backbone_state: dict):
        super().__init__()

        config = backbone_state['config']
        node_features = config['node_features']
        edge_features = config['edge_features']
        hidden_channels = config['hidden_channels']
        num_conv_layers = config['num_conv_layers']

        # Build backbone
        self.conv1 = CGConv(channels=node_features, dim=edge_features, aggr='add')
        self.lin_in = nn.Linear(node_features, hidden_channels)
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_channels, dim=edge_features, aggr='add')
            for _ in range(num_conv_layers - 1)
        ])

        # Load pretrained weights
        self.conv1.load_state_dict(backbone_state['conv1'])
        self.lin_in.load_state_dict(backbone_state['lin_in'])
        for i, conv_state in enumerate(backbone_state['convs']):
            self.convs[i].load_state_dict(conv_state)

        # Freeze backbone
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

        # Linear probe (single layer, no hidden)
        self.probe = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Frozen backbone
        with torch.no_grad():
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.lin_in(x)
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
            x = global_mean_pool(x, batch)

        # Trainable probe
        return self.probe(x)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            labels = batch.y.squeeze().cpu().numpy()

            if probs.ndim == 0:
                probs = np.array([probs])
                labels = np.array([labels])

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))

    return {'auc': auc, 'accuracy': acc, 'n': len(all_labels), 'pos': int(all_labels.sum())}


def main():
    print("=" * 60)
    print("Phase 5b: USSF Transfer Validation on DFL Open-Play")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load DFL open-play data
    print("\nLoading DFL open-play data...")
    with open(DATA_DIR / "dfl_openplay_graphs.pkl", 'rb') as f:
        data = pickle.load(f)

    graphs = data['graphs']
    labels = data['labels']
    print(f"Loaded {len(graphs)} graphs, {sum(labels)} positive ({100*sum(labels)/len(labels):.1f}%)")

    # Add labels to graphs
    for g, y in zip(graphs, labels):
        g.y = torch.tensor([y], dtype=torch.float32)

    # Split: 70% train, 15% val, 15% test
    indices = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42,
                                          stratify=[labels[i] for i in test_idx])

    train_data = [graphs[i] for i in train_idx]
    val_data = [graphs[i] for i in val_idx]
    test_data = [graphs[i] for i in test_idx]

    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Load USSF pretrained backbone
    print("\nLoading USSF pretrained backbone...")
    backbone_state = torch.load(WEIGHTS_DIR / "ussf_backbone_dense.pt", map_location=device)

    # Create model with frozen backbone + linear probe
    model = FrozenTransferModel(backbone_state).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable} trainable / {total} total (backbone frozen)")

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Class weight
    n_pos = sum(labels[i] for i in train_idx)
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    # Train linear probe
    optimizer = torch.optim.Adam(model.probe.parameters(), lr=1e-3)

    print("\nTraining linear probe (50 epochs)...")
    best_val_auc = 0
    best_state = None

    for epoch in range(1, 51):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze(), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = model.state_dict().copy()
            marker = '*'
        else:
            marker = ''

        if epoch % 10 == 0 or marker:
            print(f"  Epoch {epoch:2d}: val_auc={val_metrics['auc']:.4f} {marker}")

    # Load best and evaluate
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("RESULTS: USSF Pretrained → DFL Open-Play")
    print("=" * 60)
    print(f"Test samples: {test_metrics['n']} ({test_metrics['pos']} positive)")
    print(f"Test AUC:     {test_metrics['auc']:.4f}")
    print(f"Test Acc:     {test_metrics['accuracy']:.4f}")
    print()

    if test_metrics['auc'] > 0.55:
        print("CONCLUSION: USSF representations TRANSFER to DFL data.")
        print("The pretrained features generalize cross-dataset for shot prediction.")
    else:
        print("CONCLUSION: USSF representations DO NOT transfer to DFL data.")
        print("The corner result (0.57 AUC) is likely noise from small sample size.")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'timestamp': timestamp,
        'test_auc': test_metrics['auc'],
        'test_acc': test_metrics['accuracy'],
        'test_n': test_metrics['n'],
        'test_pos': test_metrics['pos'],
        'train_n': len(train_data),
        'val_n': len(val_data),
    }

    import json
    with open(RESULTS_DIR / f"phase5b_ussf_transfer_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: phase5b_ussf_transfer_{timestamp}.json")


if __name__ == "__main__":
    main()
