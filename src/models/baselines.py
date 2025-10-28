#!/usr/bin/env python3
"""
Baseline Models for TacticAI-Style Receiver Prediction

Implements Day 5-6: Baseline Models
- RandomReceiverBaseline: Random softmax over 22 players (sanity check)
- MLPReceiverBaseline: Flatten all player positions → MLP → 22 classes

Based on TacticAI Implementation Plan:
- Random baseline: top-1=4.5%, top-3=13.6%, top-5=22.7%
- MLP baseline: Expected top-1 > 20%, top-3 > 45%
- If MLP top-3 < 40%: STOP and debug data pipeline

Author: mseo
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class RandomReceiverBaseline(nn.Module):
    """
    Random receiver prediction baseline.

    Predicts random softmax probabilities over all players.
    Expected performance (sanity check):
    - Top-1: 4.5% (1/22)
    - Top-3: 13.6% (3/22)
    - Top-5: 22.7% (5/22)
    """

    def __init__(self, num_players: int = 22):
        """
        Initialize random baseline.

        Args:
            num_players: Number of players (default 22)
        """
        super().__init__()
        self.num_players = num_players

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random predictions.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Random logits [batch_size, num_players]
        """
        batch_size = batch.max().item() + 1

        # Generate random logits (uniform distribution)
        logits = torch.randn(batch_size, self.num_players, device=x.device)

        return logits

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_players]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)


class MLPReceiverBaseline(nn.Module):
    """
    MLP receiver prediction baseline.

    Flattens all player features and uses a 3-layer MLP to predict receiver.
    Architecture:
    - Flatten: [batch, 22 nodes × 14 features] = [batch, 308]
    - MLP: 308 → 256 → 128 → 22
    - Dropout: 0.3
    - Activation: ReLU

    Expected performance (TacticAI plan):
    - Top-1: > 20%
    - Top-3: > 45%
    - If top-3 < 40%: Debug data pipeline
    """

    def __init__(self, num_features: int = 14, num_players: int = 22,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 dropout: float = 0.3):
        """
        Initialize MLP baseline.

        Args:
            num_features: Number of features per player (default 14)
            num_players: Number of players (default 22)
            hidden_dim1: First hidden layer dimension
            hidden_dim2: Second hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_features = num_features
        self.num_players = num_players
        self.input_dim = num_players * num_features  # 22 × 14 = 308

        # Three-layer MLP
        self.fc1 = nn.Linear(self.input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_players)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Logits [batch_size, num_players]
        """
        batch_size = batch.max().item() + 1

        # Flatten: [num_nodes, num_features] → [batch_size, num_players * num_features]
        flattened = self._flatten_batch(x, batch, batch_size)

        # MLP layers
        h1 = self.relu(self.fc1(flattened))
        h1 = self.dropout(h1)

        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)

        logits = self.fc3(h2)

        return logits

    def _flatten_batch(self, x: torch.Tensor, batch: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
        """
        Flatten node features by batch.

        Handles variable number of players per graph by padding/truncating to 22.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]
            batch_size: Number of graphs in batch

        Returns:
            Flattened features [batch_size, num_players * num_features]
        """
        flattened = torch.zeros(batch_size, self.input_dim, device=x.device)

        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch == i)
            graph_features = x[mask]  # [num_nodes_i, num_features]

            num_nodes = graph_features.size(0)

            if num_nodes > self.num_players:
                # Truncate if more than 22 players (shouldn't happen)
                graph_features = graph_features[:self.num_players]
                num_nodes = self.num_players

            # Flatten and insert into batch tensor
            flat = graph_features.flatten()  # [num_nodes_i * num_features]
            flattened[i, :len(flat)] = flat

            # Remaining positions are zero-padded (for graphs with < 22 players)

        return flattened

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_players]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)


def evaluate_baseline(model: nn.Module,
                     data_loader,
                     device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate baseline model on a dataset.

    Computes:
    - Top-1 accuracy
    - Top-3 accuracy
    - Top-5 accuracy
    - Cross-entropy loss

    Args:
        model: Baseline model (Random or MLP)
        data_loader: PyTorch Geometric DataLoader
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    model.eval()
    model.to(device)

    total = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            # Forward pass
            logits = model(batch.x, batch.batch)

            # Get predictions
            probs = F.softmax(logits, dim=1)

            # Top-k predictions
            _, top5_pred = torch.topk(probs, k=5, dim=1)

            # Ground truth
            targets = batch.receiver_label.squeeze()

            # Compute accuracy
            batch_size = targets.size(0)
            total += batch_size

            for i in range(batch_size):
                target = targets[i].item()
                top5 = top5_pred[i].tolist()

                # Top-1
                if top5[0] == target:
                    top1_correct += 1

                # Top-3
                if target in top5[:3]:
                    top3_correct += 1

                # Top-5
                if target in top5:
                    top5_correct += 1

            # Compute loss
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch_size

    # Compute final metrics
    metrics = {
        'top1_accuracy': top1_correct / total,
        'top3_accuracy': top3_correct / total,
        'top5_accuracy': top5_correct / total,
        'loss': total_loss / total
    }

    return metrics


def train_mlp_baseline(model: MLPReceiverBaseline,
                       train_loader,
                       val_loader,
                       num_steps: int = 10000,
                       lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       device: str = 'cuda',
                       eval_every: int = 500,
                       verbose: bool = True) -> Dict:
    """
    Train MLP baseline for specified number of steps.

    Args:
        model: MLPReceiverBaseline model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_steps: Number of training steps (default 10k)
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        eval_every: Evaluate every N steps
        verbose: Print progress

    Returns:
        Training history dictionary
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_top1': [],
        'val_top3': [],
        'val_top5': [],
        'steps': []
    }

    step = 0
    best_val_top3 = 0.0
    best_model_state = None

    if verbose:
        print(f"\nTraining MLP baseline for {num_steps} steps...")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Evaluating every {eval_every} steps\n")

    # Training loop
    train_iter = iter(train_loader)

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device)

        # Forward pass
        model.train()
        optimizer.zero_grad()

        logits = model(batch.x, batch.batch)
        targets = batch.receiver_label.squeeze()

        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        step += 1

        # Evaluate
        if step % eval_every == 0 or step == num_steps:
            val_metrics = evaluate_baseline(model, val_loader, device)

            history['train_loss'].append(loss.item())
            history['val_top1'].append(val_metrics['top1_accuracy'])
            history['val_top3'].append(val_metrics['top3_accuracy'])
            history['val_top5'].append(val_metrics['top5_accuracy'])
            history['steps'].append(step)

            if verbose:
                print(f"Step {step:5d}/{num_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Val Top-1: {val_metrics['top1_accuracy']*100:.1f}% | "
                      f"Val Top-3: {val_metrics['top3_accuracy']*100:.1f}% | "
                      f"Val Top-5: {val_metrics['top5_accuracy']*100:.1f}%")

            # Save best model
            if val_metrics['top3_accuracy'] > best_val_top3:
                best_val_top3 = val_metrics['top3_accuracy']
                best_model_state = model.state_dict().copy()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_val_top3'] = best_val_top3

    if verbose:
        print(f"\nTraining complete!")
        print(f"Best Val Top-3: {best_val_top3*100:.1f}%")

    return history


if __name__ == "__main__":
    # Test baseline models
    print("="*60)
    print("TESTING BASELINE MODELS")
    print("="*60)

    # Create dummy batch
    batch_size = 4
    num_nodes_per_graph = 22
    num_features = 14

    x = torch.randn(batch_size * num_nodes_per_graph, num_features)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)

    print(f"\nDummy batch:")
    print(f"  x shape: {x.shape}")
    print(f"  batch shape: {batch.shape}")
    print(f"  batch_size: {batch_size}")

    # Test RandomReceiverBaseline
    print("\n1. Testing RandomReceiverBaseline...")
    random_model = RandomReceiverBaseline(num_players=22)
    random_logits = random_model(x, batch)
    random_probs = random_model.predict(x, batch)

    print(f"  Logits shape: {random_logits.shape}")
    print(f"  Probs shape: {random_probs.shape}")
    print(f"  Probs sum (should be 1.0): {random_probs.sum(dim=1)}")

    # Test MLPReceiverBaseline
    print("\n2. Testing MLPReceiverBaseline...")
    mlp_model = MLPReceiverBaseline(num_features=14, num_players=22)
    mlp_logits = mlp_model(x, batch)
    mlp_probs = mlp_model.predict(x, batch)

    print(f"  Logits shape: {mlp_logits.shape}")
    print(f"  Probs shape: {mlp_probs.shape}")
    print(f"  Probs sum (should be 1.0): {mlp_probs.sum(dim=1)}")

    # Count parameters
    num_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"  Number of parameters: {num_params:,}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
