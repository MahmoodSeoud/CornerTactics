"""Focal Loss implementation for class imbalance handling.

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

References:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    https://arxiv.org/abs/1708.02002
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Down-weights well-classified examples to focus training on hard examples.
    When gamma=0, this reduces to standard binary cross-entropy.

    Args:
        gamma: Focusing parameter (default: 2.0). Higher values increase
            focus on hard examples.
        alpha: Class weight for positive class (default: None for balanced).
            Can be a scalar or tensor of shape [2] for per-class weights.
        reduction: Reduction mode ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid) [batch_size, 1]
            targets: Binary targets {0, 1} [batch_size, 1]

        Returns:
            Focal loss (scalar if reduction is 'mean' or 'sum',
            else [batch_size, 1])
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            if self.alpha.ndim == 0:
                # Scalar alpha: weight for positive class
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Per-class alpha tensor
                alpha_t = (
                    self.alpha[1] * targets + self.alpha[0] * (1 - targets)
                )
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class FocalLossMulticlass(nn.Module):
    """Focal Loss for multi-class classification.

    Extends focal loss to multi-class setting using cross-entropy base.

    Args:
        num_classes: Number of classes
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights tensor of shape [num_classes] (default: None)
        reduction: Reduction mode ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-class focal loss.

        Args:
            logits: Raw model outputs [batch_size, num_classes]
            targets: Class indices [batch_size]

        Returns:
            Focal loss (scalar if reduction is 'mean' or 'sum',
            else [batch_size])
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Get probability of true class for each sample
        batch_size = logits.shape[0]
        p_t = probs[torch.arange(batch_size, device=logits.device), targets]

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
