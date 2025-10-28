#!/usr/bin/env python3
"""
Focal Loss implementation for handling class imbalance.
Based on: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    FL(pt) = -alpha * (1-pt)^gamma * log(pt)

    where:
    - pt is the model's estimated probability for the correct class
    - alpha is the weighting factor for class imbalance
    - gamma is the focusing parameter (typically 2)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
                   or a tensor of shape (batch_size,) for per-sample weighting
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape (N,) or (N, 1)
            targets: Ground truth labels of shape (N,) or (N, 1)
        """
        # Ensure proper shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate focal loss
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Apply focal term
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(FocalLoss):
    """
    Focal Loss with automatic class weight calculation.
    """
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            pos_weight: Weight for positive class. If None, uses 0.25
            gamma: Focusing parameter
            reduction: Reduction method
        """
        # Convert pos_weight to alpha
        if pos_weight is not None:
            alpha = 1.0 / (1.0 + pos_weight)
        else:
            alpha = 0.25

        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction)
        self.pos_weight = pos_weight


def compare_losses():
    """Compare BCE vs Focal Loss on imbalanced data"""

    # Simulate imbalanced batch
    batch_size = 100
    pos_samples = 18  # 18% positive

    # Create targets
    targets = torch.zeros(batch_size)
    targets[:pos_samples] = 1.0

    # Simulate predictions (all negative bias)
    logits = torch.randn(batch_size) - 1.0  # Bias toward negative

    # Calculate different losses
    bce_loss = nn.BCEWithLogitsLoss()
    weighted_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.5]))
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    weighted_focal = WeightedFocalLoss(pos_weight=4.5, gamma=2.0)

    print("Loss Comparison on Imbalanced Batch:")
    print(f"  Standard BCE: {bce_loss(logits, targets):.4f}")
    print(f"  Weighted BCE (4.5): {weighted_bce(logits, targets):.4f}")
    print(f"  Focal Loss: {focal(logits, targets):.4f}")
    print(f"  Weighted Focal: {weighted_focal(logits, targets):.4f}")


if __name__ == "__main__":
    # Test focal loss
    compare_losses()