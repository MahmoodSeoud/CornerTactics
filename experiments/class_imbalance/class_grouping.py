"""Class Grouping Experiments for handling severe class imbalance.

Provides utilities to group the 8 corner kick outcome classes into
coarser groupings to improve classifier performance:

- Binary: SHOT vs NO_SHOT
- Ternary: SHOT vs NO_SHOT vs PROCEDURAL
- Full: Original 8 classes

Usage:
    Run experiments with different groupings to compare performance
    and determine if class merging improves signal detection.
"""

from typing import Dict, List, Tuple, Optional, Literal, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


# Fine-grained class names in standard order (matching hierarchical_classifier)
FINE_CLASSES = [
    'GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET',  # indices 0, 1, 2
    'NOT_DANGEROUS', 'CLEARED',                    # indices 3, 4
    'FOUL', 'OFFSIDE', 'CORNER_WON',               # indices 5, 6, 7
]

# Binary grouping: SHOT vs NO_SHOT (includes procedural in NO_SHOT)
BINARY_GROUPING: Dict[str, List[str]] = {
    'SHOT': ['GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET'],
    'NO_SHOT': ['NOT_DANGEROUS', 'CLEARED', 'FOUL', 'OFFSIDE', 'CORNER_WON'],
}

# Ternary grouping: SHOT vs NO_SHOT vs PROCEDURAL
TERNARY_GROUPING: Dict[str, List[str]] = {
    'SHOT': ['GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET'],
    'NO_SHOT': ['NOT_DANGEROUS', 'CLEARED'],
    'PROCEDURAL': ['FOUL', 'OFFSIDE', 'CORNER_WON'],
}

# Mapping from fine index to grouped index
BINARY_INDEX_MAP = {
    0: 0, 1: 0, 2: 0,  # SHOT
    3: 1, 4: 1, 5: 1, 6: 1, 7: 1,  # NO_SHOT
}

TERNARY_INDEX_MAP = {
    0: 0, 1: 0, 2: 0,  # SHOT
    3: 1, 4: 1,         # NO_SHOT
    5: 2, 6: 2, 7: 2,   # PROCEDURAL
}


def create_label_mapper(
    grouping: Literal['binary', 'ternary', 'full'],
) -> Callable[[int], int]:
    """Create a function that maps fine indices to grouped indices.

    Args:
        grouping: Grouping type ('binary', 'ternary', or 'full')

    Returns:
        Function that maps fine index to grouped index
    """
    if grouping == 'binary':
        index_map = BINARY_INDEX_MAP
    elif grouping == 'ternary':
        index_map = TERNARY_INDEX_MAP
    elif grouping == 'full':
        index_map = {i: i for i in range(8)}  # Identity
    else:
        raise ValueError(f"Unknown grouping: {grouping}")

    def mapper(fine_index: int) -> int:
        return index_map[fine_index]

    return mapper


def map_labels(
    fine_labels: torch.Tensor,
    grouping: Literal['binary', 'ternary', 'full'],
) -> torch.Tensor:
    """Map batch of fine labels to grouped labels.

    Args:
        fine_labels: Tensor of fine-grained class indices [n_samples]
        grouping: Grouping type

    Returns:
        Tensor of grouped class indices [n_samples]
    """
    if grouping == 'binary':
        index_map = BINARY_INDEX_MAP
    elif grouping == 'ternary':
        index_map = TERNARY_INDEX_MAP
    elif grouping == 'full':
        return fine_labels.clone()
    else:
        raise ValueError(f"Unknown grouping: {grouping}")

    # Create mapping tensor
    mapping = torch.tensor(
        [index_map[i] for i in range(8)],
        dtype=torch.long,
        device=fine_labels.device,
    )

    return mapping[fine_labels]


def get_num_classes(grouping: Literal['binary', 'ternary', 'full']) -> int:
    """Get number of classes for a grouping.

    Args:
        grouping: Grouping type

    Returns:
        Number of classes in the grouping
    """
    if grouping == 'binary':
        return 2
    elif grouping == 'ternary':
        return 3
    elif grouping == 'full':
        return 8
    else:
        raise ValueError(f"Unknown grouping: {grouping}")


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for embeddings.

    Args:
        embedding_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: [batch_size, embedding_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        return self.net(embeddings)


class ClassGroupingExperiment:
    """Experiment for evaluating a specific class grouping.

    Args:
        embedding_dim: Input embedding dimension
        grouping: Grouping type ('binary', 'ternary', 'full')
        hidden_dim: Hidden layer dimension
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to use
    """

    def __init__(
        self,
        embedding_dim: int,
        grouping: Literal['binary', 'ternary', 'full'] = 'binary',
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.grouping = grouping
        self.num_classes = get_num_classes(grouping)
        self.device = device or torch.device('cpu')

        self.model = SimpleClassifier(
            embedding_dim=embedding_dim,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _prepare_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Map fine labels to grouped labels."""
        return map_labels(fine_labels, self.grouping)

    def train(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 50,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train the classifier.

        Args:
            train_embeddings: Training embeddings
            train_labels: Fine-grained training labels
            val_embeddings: Validation embeddings
            val_labels: Fine-grained validation labels
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            Training history
        """
        train_embeddings = train_embeddings.to(self.device)
        val_embeddings = val_embeddings.to(self.device)
        train_labels_grouped = self._prepare_labels(train_labels).to(self.device)
        val_labels_grouped = self._prepare_labels(val_labels).to(self.device)

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(train_embeddings)
            loss = self.criterion(logits, train_labels_grouped)
            loss.backward()
            self.optimizer.step()
            history['train_loss'].append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_embeddings)
                val_loss = self.criterion(val_logits, val_labels_grouped)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == val_labels_grouped).float().mean().item()

            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return history

    def evaluate(
        self,
        embeddings: torch.Tensor,
        fine_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate the classifier.

        Args:
            embeddings: Test embeddings
            fine_labels: Fine-grained test labels

        Returns:
            Dictionary with accuracy and AUC
        """
        self.model.eval()
        embeddings = embeddings.to(self.device)
        grouped_labels = self._prepare_labels(fine_labels).to(self.device)

        with torch.no_grad():
            logits = self.model(embeddings)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        # Accuracy
        accuracy = (preds == grouped_labels).float().mean().item()

        # AUC (one-vs-rest for multiclass)
        probs_np = probs.cpu().numpy()
        labels_np = grouped_labels.cpu().numpy()

        try:
            if self.num_classes == 2:
                auc = roc_auc_score(labels_np, probs_np[:, 1])
            else:
                auc = roc_auc_score(
                    labels_np, probs_np, multi_class='ovr', average='weighted'
                )
        except ValueError:
            # If only one class present
            auc = 0.5

        return {'accuracy': accuracy, 'auc': auc}


def run_all_grouping_experiments(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = 50,
    verbose: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Run experiments with all class groupings.

    Args:
        train_embeddings: Training embeddings
        train_labels: Fine-grained training labels
        val_embeddings: Validation embeddings
        val_labels: Fine-grained validation labels
        test_embeddings: Test embeddings
        test_labels: Fine-grained test labels
        epochs: Number of training epochs
        verbose: Print progress
        device: Device to use

    Returns:
        Dictionary mapping grouping name to metrics
    """
    embedding_dim = train_embeddings.shape[1]
    results = {}

    for grouping in ['binary', 'ternary', 'full']:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running {grouping.upper()} grouping experiment")
            print(f"{'='*50}")

        experiment = ClassGroupingExperiment(
            embedding_dim=embedding_dim,
            grouping=grouping,
            device=device,
        )

        experiment.train(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            epochs=epochs,
            verbose=verbose,
        )

        metrics = experiment.evaluate(test_embeddings, test_labels)
        results[grouping] = metrics

        if verbose:
            print(f"\n{grouping.upper()} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")

    return results
