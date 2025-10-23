#!/usr/bin/env python3
"""
Training Utilities for Corner Kick GNN

Helper functions for model training including:
- Metrics computation
- Early stopping
- Model checkpointing
- Training/validation loops

Author: mseo
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)
import warnings
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation metric and stops training when it stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0001,
                 mode: str = 'max', verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current validation score

        Returns:
            True if should stop training
        """
        if self.mode == 'max':
            score = score
        else:
            score = -score

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"Initial score: {score:.4f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Score improved: {self.best_score:.4f} -> {score:.4f}")
            self.best_score = score
            self.counter = 0

        return self.early_stop


class MetricsComputer:
    """Compute various metrics for binary classification."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics computer.

        Args:
            threshold: Decision threshold for binary classification
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor,
              probs: Optional[torch.Tensor] = None):
        """
        Add batch predictions and labels.

        Args:
            preds: Binary predictions [batch_size]
            labels: True labels [batch_size]
            probs: Probability predictions [batch_size]
        """
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0)
        }

        # AUC if probabilities available
        if self.all_probs:
            probs = np.array(self.all_probs)
            try:
                metrics['auc_roc'] = roc_auc_score(labels, probs)
            except ValueError:
                # Can happen if only one class in batch
                metrics['auc_roc'] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

        return metrics


def train_epoch(model: nn.Module, loader, optimizer,
               criterion, device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: GNN model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    metrics = MetricsComputer()

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        probs = torch.sigmoid(out.squeeze())
        preds = (probs > 0.5).float()
        metrics.update(preds, batch.y.squeeze(), probs)

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = total_loss / len(loader.dataset)

    return epoch_metrics


@torch.no_grad()
def validate_epoch(model: nn.Module, loader, criterion,
                  device: torch.device) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: GNN model
        loader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    metrics = MetricsComputer()

    progress_bar = tqdm(loader, desc="Validating", leave=False)
    for batch in progress_bar:
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        probs = torch.sigmoid(out.squeeze())
        preds = (probs > 0.5).float()
        metrics.update(preds, batch.y.squeeze(), probs)

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = total_loss / len(loader.dataset)

    return epoch_metrics


def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                   metrics: Dict[str, float], filepath: str):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_checkpoint(model: nn.Module, optimizer, filepath: str) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint

    Returns:
        Checkpoint dictionary with epoch and metrics
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {filepath}")
    return checkpoint


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for printing (e.g., "Train", "Val")
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")

    # Main metrics
    main_metrics = ['loss', 'accuracy', 'auc_roc', 'precision', 'recall', 'f1']
    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric:12s}: {metrics[metric]:.4f}")

    # Confusion matrix if available
    if 'true_positives' in metrics:
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"    FN: {metrics['false_negatives']:4d}  TN: {metrics['true_negatives']:4d}")


def compute_class_weights(loader) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.

    Args:
        loader: Data loader

    Returns:
        Class weights tensor
    """
    all_labels = []
    for batch in loader:
        all_labels.extend(batch.y.cpu().numpy())

    labels = np.array(all_labels)
    n_samples = len(labels)
    n_positive = labels.sum()
    n_negative = n_samples - n_positive

    # Compute weights inversely proportional to class frequency
    weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
    weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0

    print(f"Class weights - Negative: {weight_negative:.2f}, Positive: {weight_positive:.2f}")

    return torch.FloatTensor([weight_negative, weight_positive])


def focal_loss(preds: torch.Tensor, targets: torch.Tensor,
              alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance.

    Args:
        preds: Predictions (logits)
        targets: True labels
        alpha: Weight for positive class
        gamma: Focusing parameter

    Returns:
        Focal loss value
    """
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")

    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping triggered at epoch {i+1}")
            break

    # Test MetricsComputer
    metrics_computer = MetricsComputer()
    preds = torch.tensor([1, 0, 1, 1, 0])
    labels = torch.tensor([1, 0, 0, 1, 1])
    probs = torch.tensor([0.9, 0.1, 0.7, 0.8, 0.3])

    metrics_computer.update(preds, labels, probs)
    metrics = metrics_computer.compute()
    print_metrics(metrics)

    print("\n✅ Training utilities tests passed!")