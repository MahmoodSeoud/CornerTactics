"""Tests for Hierarchical Classification.

TDD: Write tests first, then implement to make them pass.

Hierarchical classification for corner kick outcomes:
- Level 1: SHOT vs NO_SHOT vs PROCEDURAL
- Level 2: Fine-grained within each group
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestClassHierarchy:
    """Test class hierarchy definition."""

    def test_hierarchy_definition_exists(self):
        """CLASS_HIERARCHY should define the label groupings."""
        from experiments.class_imbalance.hierarchical_classifier import CLASS_HIERARCHY

        assert 'SHOT' in CLASS_HIERARCHY
        assert 'NO_SHOT' in CLASS_HIERARCHY
        assert 'PROCEDURAL' in CLASS_HIERARCHY

    def test_shot_classes(self):
        """SHOT should contain goal, shot_on_target, shot_off_target."""
        from experiments.class_imbalance.hierarchical_classifier import CLASS_HIERARCHY

        assert 'GOAL' in CLASS_HIERARCHY['SHOT']
        assert 'SHOT_ON_TARGET' in CLASS_HIERARCHY['SHOT']
        assert 'SHOT_OFF_TARGET' in CLASS_HIERARCHY['SHOT']

    def test_no_shot_classes(self):
        """NO_SHOT should contain not_dangerous, cleared."""
        from experiments.class_imbalance.hierarchical_classifier import CLASS_HIERARCHY

        assert 'NOT_DANGEROUS' in CLASS_HIERARCHY['NO_SHOT']
        assert 'CLEARED' in CLASS_HIERARCHY['NO_SHOT']

    def test_procedural_classes(self):
        """PROCEDURAL should contain foul, offside, corner_won."""
        from experiments.class_imbalance.hierarchical_classifier import CLASS_HIERARCHY

        assert 'FOUL' in CLASS_HIERARCHY['PROCEDURAL']
        assert 'OFFSIDE' in CLASS_HIERARCHY['PROCEDURAL']
        assert 'CORNER_WON' in CLASS_HIERARCHY['PROCEDURAL']


class TestLabelMapper:
    """Test label mapping utilities."""

    def test_map_to_coarse_labels(self):
        """Should map fine-grained labels to coarse (level 1)."""
        from experiments.class_imbalance.hierarchical_classifier import map_to_coarse_labels

        fine_labels = ['GOAL', 'NOT_DANGEROUS', 'FOUL', 'SHOT_ON_TARGET']
        coarse_labels = map_to_coarse_labels(fine_labels)

        assert coarse_labels == ['SHOT', 'NO_SHOT', 'PROCEDURAL', 'SHOT']

    def test_map_to_coarse_with_indices(self):
        """Should map label indices to coarse indices."""
        from experiments.class_imbalance.hierarchical_classifier import (
            map_to_coarse_indices, FINE_TO_COARSE_MAP
        )

        # Assuming standard ordering
        fine_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # All 8 classes
        coarse_indices = map_to_coarse_indices(fine_indices)

        # All indices should be 0, 1, or 2
        assert coarse_indices.min() >= 0
        assert coarse_indices.max() <= 2


class TestHierarchicalModel:
    """Test hierarchical classification model."""

    def test_instantiation(self):
        """HierarchicalClassifier should instantiate."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalClassifier

        model = HierarchicalClassifier(
            in_channels=64,  # embedding dimension
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        assert model is not None

    def test_forward_returns_both_predictions(self):
        """Forward should return coarse and fine predictions."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalClassifier

        model = HierarchicalClassifier(
            in_channels=64,
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        embeddings = torch.randn(10, 64)
        coarse_logits, fine_logits = model(embeddings)

        assert coarse_logits.shape == (10, 3)
        assert fine_logits.shape == (10, 8)

    def test_gradients_flow(self):
        """Gradients should flow through both heads."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalClassifier

        model = HierarchicalClassifier(
            in_channels=64,
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        embeddings = torch.randn(10, 64, requires_grad=True)
        coarse_logits, fine_logits = model(embeddings)

        loss = coarse_logits.sum() + fine_logits.sum()
        loss.backward()

        assert embeddings.grad is not None


class TestHierarchicalLoss:
    """Test hierarchical classification loss."""

    def test_loss_instantiation(self):
        """HierarchicalLoss should instantiate."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalLoss

        loss_fn = HierarchicalLoss(
            coarse_weight=1.0,
            fine_weight=1.0,
        )

        assert loss_fn is not None

    def test_loss_output_scalar(self):
        """Loss should return scalar."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalLoss

        loss_fn = HierarchicalLoss()

        coarse_logits = torch.randn(10, 3)
        fine_logits = torch.randn(10, 8)
        coarse_targets = torch.randint(0, 3, (10,))
        fine_targets = torch.randint(0, 8, (10,))

        loss = loss_fn(coarse_logits, fine_logits, coarse_targets, fine_targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_loss_with_class_weights(self):
        """Loss should accept class weights."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalLoss

        coarse_weights = torch.tensor([1.0, 1.0, 2.0])
        fine_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0])

        loss_fn = HierarchicalLoss(
            coarse_weights=coarse_weights,
            fine_weights=fine_weights,
        )

        coarse_logits = torch.randn(10, 3)
        fine_logits = torch.randn(10, 8)
        coarse_targets = torch.randint(0, 3, (10,))
        fine_targets = torch.randint(0, 8, (10,))

        loss = loss_fn(coarse_logits, fine_logits, coarse_targets, fine_targets)

        assert loss >= 0


class TestHierarchicalTrainer:
    """Test hierarchical classifier training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        embeddings = torch.randn(100, 64)
        coarse_labels = torch.randint(0, 3, (100,))
        fine_labels = torch.randint(0, 8, (100,))
        return embeddings, coarse_labels, fine_labels

    def test_trainer_instantiation(self, sample_data):
        """HierarchicalTrainer should instantiate."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalTrainer

        embeddings, coarse_labels, fine_labels = sample_data

        trainer = HierarchicalTrainer(
            in_channels=64,
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        assert trainer is not None

    def test_train_epoch(self, sample_data):
        """Should complete one training epoch."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalTrainer

        embeddings, coarse_labels, fine_labels = sample_data

        trainer = HierarchicalTrainer(
            in_channels=64,
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        loss = trainer.train_epoch(embeddings, coarse_labels, fine_labels)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_predict(self, sample_data):
        """Should return predictions for coarse and fine levels."""
        from experiments.class_imbalance.hierarchical_classifier import HierarchicalTrainer

        embeddings, _, _ = sample_data

        trainer = HierarchicalTrainer(
            in_channels=64,
            num_coarse_classes=3,
            num_fine_classes=8,
        )

        coarse_preds, fine_preds = trainer.predict(embeddings)

        assert coarse_preds.shape == (100,)
        assert fine_preds.shape == (100,)
        assert coarse_preds.min() >= 0
        assert coarse_preds.max() <= 2
        assert fine_preds.min() >= 0
        assert fine_preds.max() <= 7
