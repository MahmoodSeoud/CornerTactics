"""Training module for GNN baseline.

Provides training utilities including:
- Data loading with train/val/test splits
- Trainer class with early stopping
- Class-weighted loss for imbalanced data
- Checkpointing and reproducibility
"""

import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from experiments.gnn_baseline.graph_construction import corner_to_graph, build_graph_dataset
from experiments.gnn_baseline.models import create_model


def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute inverse frequency class weights.

    Args:
        labels: List of class labels (0 or 1)

    Returns:
        Dictionary mapping class to weight
    """
    labels = np.array(labels)
    counts = np.bincount(labels)
    total = len(labels)

    weights = {}
    for cls in range(len(counts)):
        if counts[cls] > 0:
            weights[cls] = total / (len(counts) * counts[cls])
        else:
            weights[cls] = 1.0

    return weights


def create_dataloaders(
    corners: List[Dict[str, Any]],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = 32,
    edge_type: str = 'knn',
    k: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from corner data.

    Args:
        corners: List of corner dictionaries
        train_indices: Indices for training set
        val_indices: Indices for validation set
        test_indices: Indices for test set
        batch_size: Batch size for dataloaders
        edge_type: Edge construction method ('knn' or 'full')
        k: Number of neighbors for k-NN

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Build graphs for each split
    train_corners = [corners[i] for i in train_indices]
    val_corners = [corners[i] for i in val_indices]
    test_corners = [corners[i] for i in test_indices]

    train_graphs = build_graph_dataset(train_corners, edge_type=edge_type, k=k)
    val_graphs = build_graph_dataset(val_corners, edge_type=edge_type, k=k)
    test_graphs = build_graph_dataset(test_corners, edge_type=edge_type, k=k)

    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class Trainer:
    """Trainer for GNN models on corner outcome prediction."""

    def __init__(
        self,
        corners: List[Dict[str, Any]],
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        model_name: str = 'graphsage',
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        edge_type: str = 'knn',
        k: int = 5,
        use_class_weights: bool = False,
        device: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            corners: List of corner dictionaries
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            model_name: GNN architecture ('gat', 'graphsage', 'mpnn')
            hidden_channels: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            edge_type: Edge construction method
            k: Number of neighbors for k-NN
            use_class_weights: Whether to use class-weighted loss
            device: Device to use (auto-detect if None)
        """
        self.corners = corners
        self.model_name = model_name
        self.patience = patience
        self.use_class_weights = use_class_weights

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            corners, train_indices, val_indices, test_indices,
            batch_size=batch_size, edge_type=edge_type, k=k
        )

        # Get feature dimensions from first graph
        sample_graph = next(iter(self.train_loader))
        in_channels = sample_graph.x.shape[1]
        edge_channels = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 2

        # Create model
        self.model = create_model(
            name=model_name,
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss function
        if use_class_weights:
            train_labels = [corners[i]['shot_outcome'] for i in train_indices]
            weights = compute_class_weights(train_labels)
            pos_weight = torch.tensor([weights[1] / weights[0]], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # We'll apply sigmoid separately for predictions
            self._use_logits = True
        else:
            self.criterion = nn.BCELoss()
            self._use_logits = False

    def train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            if self.model_name == 'mpnn':
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)

            # Compute loss
            if self._use_logits:
                # For BCEWithLogitsLoss, we need logits (pre-sigmoid)
                # But our model outputs sigmoid, so convert back
                out_logits = torch.log(out / (1 - out + 1e-7))
                loss = self.criterion(out_logits.squeeze(), batch.y.squeeze())
            else:
                loss = self.criterion(out.squeeze(), batch.y.squeeze())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> Tuple[float, float]:
        """Evaluate on validation set.

        Returns:
            Tuple of (validation loss, validation AUC)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                if self.model_name == 'mpnn':
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    out = self.model(batch.x, batch.edge_index, batch.batch)

                if self._use_logits:
                    out_logits = torch.log(out / (1 - out + 1e-7))
                    loss = self.criterion(out_logits.squeeze(), batch.y.squeeze())
                else:
                    loss = self.criterion(out.squeeze(), batch.y.squeeze())

                total_loss += loss.item()
                num_batches += 1

                all_preds.extend(out.squeeze().cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())

        avg_loss = total_loss / num_batches

        # Compute AUC
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            # If only one class present
            auc = 0.5

        return avg_loss, auc

    def fit(
        self,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            epochs: Maximum number of epochs
            verbose: Whether to print progress

        Returns:
            Training history with losses and AUC scores
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
        }

        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def save_checkpoint(self, path: Path):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
