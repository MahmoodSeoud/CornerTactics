"""Tests for Focal Loss implementation.

TDD: Write tests first, then implement to make them pass.

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- Reduces loss for well-classified examples
- Focuses training on hard examples
"""

import pytest
import torch
import torch.nn as nn


class TestFocalLossInstantiation:
    """Test Focal Loss creation."""

    def test_instantiation_default_params(self):
        """FocalLoss should instantiate with default parameters."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss()
        assert loss_fn is not None

    def test_instantiation_custom_gamma(self):
        """FocalLoss should accept custom gamma parameter."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=3.0)
        assert loss_fn.gamma == 3.0

    def test_instantiation_custom_alpha(self):
        """FocalLoss should accept custom alpha (class weights)."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(alpha=torch.tensor([0.25, 0.75]))
        assert loss_fn.alpha is not None

    def test_is_nn_module(self):
        """FocalLoss should be a PyTorch nn.Module."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss()
        assert isinstance(loss_fn, nn.Module)


class TestFocalLossBinaryClassification:
    """Test Focal Loss for binary classification."""

    def test_output_is_scalar(self):
        """Loss should return a scalar tensor."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0  # scalar

    def test_output_is_non_negative(self):
        """Loss should always be non-negative."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(100, 1)
        targets = torch.randint(0, 2, (100, 1)).float()

        loss = loss_fn(logits, targets)

        assert loss >= 0

    def test_perfect_predictions_low_loss(self):
        """Perfect predictions should have lower loss than bad predictions."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)

        # Perfect predictions (high confidence correct)
        logits_perfect = torch.tensor([[10.0], [10.0], [-10.0], [-10.0]])
        targets = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        loss_perfect = loss_fn(logits_perfect, targets)

        # Bad predictions (wrong)
        logits_bad = torch.tensor([[-10.0], [-10.0], [10.0], [10.0]])
        loss_bad = loss_fn(logits_bad, targets)

        assert loss_perfect < loss_bad

    def test_well_classified_gets_downweighted(self):
        """Well-classified examples should contribute less to loss than BCE."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        # Compare focal loss (gamma=2) with BCE (gamma=0)
        focal_loss = FocalLoss(gamma=2.0, reduction='none')
        bce_loss = FocalLoss(gamma=0.0, reduction='none')  # gamma=0 is BCE

        # Well-classified example (high confidence correct)
        logits = torch.tensor([[5.0]])  # p ~ 0.993 for class 1
        targets = torch.tensor([[1.0]])

        fl = focal_loss(logits, targets)
        bce = bce_loss(logits, targets)

        # Focal loss should be smaller for well-classified examples
        assert fl < bce


class TestFocalLossMulticlass:
    """Test Focal Loss for multi-class classification."""

    def test_multiclass_instantiation(self):
        """FocalLoss should work with multi-class targets."""
        from experiments.class_imbalance.focal_loss import FocalLossMulticlass

        loss_fn = FocalLossMulticlass(num_classes=8)
        assert loss_fn is not None

    def test_multiclass_output_scalar(self):
        """Multiclass focal loss should return scalar."""
        from experiments.class_imbalance.focal_loss import FocalLossMulticlass

        loss_fn = FocalLossMulticlass(num_classes=8, gamma=2.0)
        logits = torch.randn(10, 8)  # 10 samples, 8 classes
        targets = torch.randint(0, 8, (10,))

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0

    def test_multiclass_with_class_weights(self):
        """Multiclass focal loss should accept class weights."""
        from experiments.class_imbalance.focal_loss import FocalLossMulticlass

        weights = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2])
        loss_fn = FocalLossMulticlass(num_classes=8, alpha=weights)
        logits = torch.randn(10, 8)
        targets = torch.randint(0, 8, (10,))

        loss = loss_fn(logits, targets)

        assert loss >= 0


class TestFocalLossGradients:
    """Test gradient flow through Focal Loss."""

    def test_binary_gradients_flow(self):
        """Gradients should flow through binary focal loss."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(10, 1, requires_grad=True)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None

    def test_multiclass_gradients_flow(self):
        """Gradients should flow through multiclass focal loss."""
        from experiments.class_imbalance.focal_loss import FocalLossMulticlass

        loss_fn = FocalLossMulticlass(num_classes=8)
        logits = torch.randn(10, 8, requires_grad=True)
        targets = torch.randint(0, 8, (10,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None


class TestFocalLossReduction:
    """Test reduction modes for Focal Loss."""

    def test_reduction_none(self):
        """reduction='none' should return per-sample losses."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(reduction='none')
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == (10, 1)

    def test_reduction_mean(self):
        """reduction='mean' should return mean of losses."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(reduction='mean')
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0

    def test_reduction_sum(self):
        """reduction='sum' should return sum of losses."""
        from experiments.class_imbalance.focal_loss import FocalLoss

        loss_fn = FocalLoss(reduction='sum')
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
