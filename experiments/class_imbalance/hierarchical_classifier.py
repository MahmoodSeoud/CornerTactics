"""Hierarchical Classification for corner kick outcomes.

Implements a two-level hierarchical classifier:
- Level 1 (coarse): SHOT vs NO_SHOT vs PROCEDURAL
- Level 2 (fine): Original 8 classes

This approach can help with severe class imbalance by first predicting
a coarse category, then refining within that category.
"""

from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


# Class hierarchy definition
CLASS_HIERARCHY: Dict[str, List[str]] = {
    'SHOT': ['GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET'],
    'NO_SHOT': ['NOT_DANGEROUS', 'CLEARED'],
    'PROCEDURAL': ['FOUL', 'OFFSIDE', 'CORNER_WON'],
}

# Fine-grained class names in standard order
FINE_CLASSES = [
    'GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET',  # SHOT (0, 1, 2)
    'NOT_DANGEROUS', 'CLEARED',                    # NO_SHOT (3, 4)
    'FOUL', 'OFFSIDE', 'CORNER_WON',               # PROCEDURAL (5, 6, 7)
]

COARSE_CLASSES = ['SHOT', 'NO_SHOT', 'PROCEDURAL']

# Mapping from fine index to coarse index
FINE_TO_COARSE_MAP = {
    0: 0, 1: 0, 2: 0,  # GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET -> SHOT
    3: 1, 4: 1,         # NOT_DANGEROUS, CLEARED -> NO_SHOT
    5: 2, 6: 2, 7: 2,   # FOUL, OFFSIDE, CORNER_WON -> PROCEDURAL
}


def map_to_coarse_labels(fine_labels: List[str]) -> List[str]:
    """Map fine-grained labels to coarse labels.

    Args:
        fine_labels: List of fine-grained class names

    Returns:
        List of coarse class names
    """
    coarse_labels = []
    for label in fine_labels:
        for coarse, fine_list in CLASS_HIERARCHY.items():
            if label in fine_list:
                coarse_labels.append(coarse)
                break
    return coarse_labels


def map_to_coarse_indices(fine_indices: torch.Tensor) -> torch.Tensor:
    """Map fine-grained indices to coarse indices.

    Args:
        fine_indices: Tensor of fine-grained class indices [0-7]

    Returns:
        Tensor of coarse class indices [0-2]
    """
    mapping = torch.tensor(
        [FINE_TO_COARSE_MAP[i] for i in range(8)],
        dtype=torch.long,
        device=fine_indices.device,
    )
    return mapping[fine_indices]


class HierarchicalClassifier(nn.Module):
    """Two-level hierarchical classifier.

    Predicts both coarse and fine-grained labels from embeddings.

    Args:
        in_channels: Input embedding dimension
        num_coarse_classes: Number of coarse classes (default: 3)
        num_fine_classes: Number of fine classes (default: 8)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        num_coarse_classes: int = 3,
        num_fine_classes: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Coarse classifier head
        self.coarse_head = nn.Linear(hidden_dim, num_coarse_classes)

        # Fine classifier head
        self.fine_head = nn.Linear(hidden_dim, num_fine_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both coarse and fine predictions.

        Args:
            embeddings: Input embeddings [batch_size, in_channels]

        Returns:
            Tuple of (coarse_logits, fine_logits)
            - coarse_logits: [batch_size, num_coarse_classes]
            - fine_logits: [batch_size, num_fine_classes]
        """
        h = self.encoder(embeddings)
        coarse_logits = self.coarse_head(h)
        fine_logits = self.fine_head(h)
        return coarse_logits, fine_logits


class HierarchicalLoss(nn.Module):
    """Combined loss for hierarchical classification.

    Combines cross-entropy losses for coarse and fine predictions.

    Args:
        coarse_weight: Weight for coarse loss (default: 1.0)
        fine_weight: Weight for fine loss (default: 1.0)
        coarse_weights: Class weights for coarse classes (default: None)
        fine_weights: Class weights for fine classes (default: None)
    """

    def __init__(
        self,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        coarse_weights: Optional[torch.Tensor] = None,
        fine_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_weights)
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_weights)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        coarse_targets: torch.Tensor,
        fine_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined hierarchical loss.

        Args:
            coarse_logits: [batch_size, num_coarse_classes]
            fine_logits: [batch_size, num_fine_classes]
            coarse_targets: [batch_size] coarse class indices
            fine_targets: [batch_size] fine class indices

        Returns:
            Combined loss scalar
        """
        coarse_loss = self.coarse_criterion(coarse_logits, coarse_targets)
        fine_loss = self.fine_criterion(fine_logits, fine_targets)

        return self.coarse_weight * coarse_loss + self.fine_weight * fine_loss


class HierarchicalTrainer:
    """Trainer for hierarchical classifier.

    Args:
        in_channels: Input embedding dimension
        num_coarse_classes: Number of coarse classes
        num_fine_classes: Number of fine classes
        hidden_dim: Hidden layer dimension
        lr: Learning rate
        weight_decay: L2 regularization
        coarse_weight: Weight for coarse loss
        fine_weight: Weight for fine loss
        device: Device to train on
    """

    def __init__(
        self,
        in_channels: int,
        num_coarse_classes: int = 3,
        num_fine_classes: int = 8,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        coarse_weights: Optional[torch.Tensor] = None,
        fine_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')

        self.model = HierarchicalClassifier(
            in_channels=in_channels,
            num_coarse_classes=num_coarse_classes,
            num_fine_classes=num_fine_classes,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.criterion = HierarchicalLoss(
            coarse_weight=coarse_weight,
            fine_weight=fine_weight,
            coarse_weights=coarse_weights,
            fine_weights=fine_weights,
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_epoch(
        self,
        embeddings: torch.Tensor,
        coarse_labels: torch.Tensor,
        fine_labels: torch.Tensor,
    ) -> float:
        """Train for one epoch.

        Args:
            embeddings: [n_samples, in_channels]
            coarse_labels: [n_samples]
            fine_labels: [n_samples]

        Returns:
            Training loss
        """
        self.model.train()
        embeddings = embeddings.to(self.device)
        coarse_labels = coarse_labels.to(self.device)
        fine_labels = fine_labels.to(self.device)

        self.optimizer.zero_grad()
        coarse_logits, fine_logits = self.model(embeddings)
        loss = self.criterion(coarse_logits, fine_logits, coarse_labels, fine_labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions.

        Args:
            embeddings: [n_samples, in_channels]

        Returns:
            Tuple of (coarse_predictions, fine_predictions)
        """
        self.model.eval()
        embeddings = embeddings.to(self.device)

        with torch.no_grad():
            coarse_logits, fine_logits = self.model(embeddings)
            coarse_preds = coarse_logits.argmax(dim=1)
            fine_preds = fine_logits.argmax(dim=1)

        return coarse_preds.cpu(), fine_preds.cpu()

    def evaluate(
        self,
        embeddings: torch.Tensor,
        coarse_labels: torch.Tensor,
        fine_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate model accuracy.

        Args:
            embeddings: [n_samples, in_channels]
            coarse_labels: [n_samples]
            fine_labels: [n_samples]

        Returns:
            Dictionary with coarse_accuracy, fine_accuracy, loss
        """
        self.model.eval()
        embeddings = embeddings.to(self.device)
        coarse_labels = coarse_labels.to(self.device)
        fine_labels = fine_labels.to(self.device)

        with torch.no_grad():
            coarse_logits, fine_logits = self.model(embeddings)
            loss = self.criterion(
                coarse_logits, fine_logits, coarse_labels, fine_labels
            )

            coarse_preds = coarse_logits.argmax(dim=1)
            fine_preds = fine_logits.argmax(dim=1)

            coarse_acc = (coarse_preds == coarse_labels).float().mean().item()
            fine_acc = (fine_preds == fine_labels).float().mean().item()

        return {
            'coarse_accuracy': coarse_acc,
            'fine_accuracy': fine_acc,
            'loss': loss.item(),
        }
