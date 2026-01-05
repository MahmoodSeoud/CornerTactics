"""Tests for Class Grouping Experiments.

TDD: Write tests first, then implement to make them pass.

Class grouping experiments to handle severe class imbalance:
- Binary: SHOT vs NO_SHOT
- Ternary: SHOT vs NO_SHOT vs PROCEDURAL
- Original 8-class
"""

import pytest
import torch
import numpy as np


class TestClassGroupings:
    """Test class grouping definitions."""

    def test_binary_grouping(self):
        """Binary grouping should map 8 classes to 2."""
        from experiments.class_imbalance.class_grouping import BINARY_GROUPING

        assert len(BINARY_GROUPING) == 2
        assert 'SHOT' in BINARY_GROUPING
        assert 'NO_SHOT' in BINARY_GROUPING

    def test_binary_shot_includes_goal(self):
        """Binary SHOT should include goal and shots."""
        from experiments.class_imbalance.class_grouping import BINARY_GROUPING

        shot_classes = BINARY_GROUPING['SHOT']
        assert 'GOAL' in shot_classes
        assert 'SHOT_ON_TARGET' in shot_classes
        assert 'SHOT_OFF_TARGET' in shot_classes

    def test_binary_no_shot_includes_rest(self):
        """Binary NO_SHOT should include everything else."""
        from experiments.class_imbalance.class_grouping import BINARY_GROUPING

        no_shot_classes = BINARY_GROUPING['NO_SHOT']
        assert 'NOT_DANGEROUS' in no_shot_classes
        assert 'CLEARED' in no_shot_classes
        assert 'FOUL' in no_shot_classes
        assert 'OFFSIDE' in no_shot_classes
        assert 'CORNER_WON' in no_shot_classes

    def test_ternary_grouping(self):
        """Ternary grouping should map 8 classes to 3."""
        from experiments.class_imbalance.class_grouping import TERNARY_GROUPING

        assert len(TERNARY_GROUPING) == 3
        assert 'SHOT' in TERNARY_GROUPING
        assert 'NO_SHOT' in TERNARY_GROUPING
        assert 'PROCEDURAL' in TERNARY_GROUPING


class TestGroupingMapper:
    """Test label grouping utilities."""

    def test_create_label_mapper_binary(self):
        """Should create mapper for binary grouping."""
        from experiments.class_imbalance.class_grouping import create_label_mapper

        mapper = create_label_mapper('binary')
        assert mapper is not None

    def test_create_label_mapper_ternary(self):
        """Should create mapper for ternary grouping."""
        from experiments.class_imbalance.class_grouping import create_label_mapper

        mapper = create_label_mapper('ternary')
        assert mapper is not None

    def test_binary_mapper_maps_indices(self):
        """Binary mapper should map fine indices to binary."""
        from experiments.class_imbalance.class_grouping import create_label_mapper

        mapper = create_label_mapper('binary')

        # Test some mappings
        # GOAL (0) -> SHOT (0)
        assert mapper(0) == 0
        # NOT_DANGEROUS (3) -> NO_SHOT (1)
        assert mapper(3) == 1
        # FOUL (5) -> NO_SHOT (1)
        assert mapper(5) == 1

    def test_ternary_mapper_maps_indices(self):
        """Ternary mapper should map fine indices to ternary."""
        from experiments.class_imbalance.class_grouping import create_label_mapper

        mapper = create_label_mapper('ternary')

        # GOAL (0) -> SHOT (0)
        assert mapper(0) == 0
        # NOT_DANGEROUS (3) -> NO_SHOT (1)
        assert mapper(3) == 1
        # FOUL (5) -> PROCEDURAL (2)
        assert mapper(5) == 2

    def test_batch_mapping(self):
        """Should map batch of fine labels to grouped labels."""
        from experiments.class_imbalance.class_grouping import map_labels

        fine_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        binary_labels = map_labels(fine_labels, grouping='binary')
        ternary_labels = map_labels(fine_labels, grouping='ternary')

        assert binary_labels.shape == fine_labels.shape
        assert ternary_labels.shape == fine_labels.shape

        # Binary should have only 0 and 1
        assert binary_labels.max() <= 1
        assert binary_labels.min() >= 0

        # Ternary should have 0, 1, 2
        assert ternary_labels.max() <= 2
        assert ternary_labels.min() >= 0


class TestClassGroupingExperiment:
    """Test running grouping experiments."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings and labels."""
        embeddings = torch.randn(100, 64)
        fine_labels = torch.randint(0, 8, (100,))
        return embeddings, fine_labels

    def test_experiment_instantiation(self):
        """ClassGroupingExperiment should instantiate."""
        from experiments.class_imbalance.class_grouping import ClassGroupingExperiment

        experiment = ClassGroupingExperiment(
            embedding_dim=64,
            grouping='binary',
        )

        assert experiment is not None

    def test_experiment_train(self, sample_embeddings):
        """Experiment should train on embeddings."""
        from experiments.class_imbalance.class_grouping import ClassGroupingExperiment

        embeddings, fine_labels = sample_embeddings

        experiment = ClassGroupingExperiment(
            embedding_dim=64,
            grouping='binary',
        )

        history = experiment.train(
            train_embeddings=embeddings[:80],
            train_labels=fine_labels[:80],
            val_embeddings=embeddings[80:],
            val_labels=fine_labels[80:],
            epochs=2,
        )

        assert 'train_loss' in history
        assert 'val_loss' in history

    def test_experiment_evaluate(self, sample_embeddings):
        """Experiment should evaluate and return metrics."""
        from experiments.class_imbalance.class_grouping import ClassGroupingExperiment

        embeddings, fine_labels = sample_embeddings

        experiment = ClassGroupingExperiment(
            embedding_dim=64,
            grouping='binary',
        )

        # Train first
        experiment.train(
            train_embeddings=embeddings[:80],
            train_labels=fine_labels[:80],
            val_embeddings=embeddings[80:],
            val_labels=fine_labels[80:],
            epochs=2,
        )

        metrics = experiment.evaluate(embeddings[80:], fine_labels[80:])

        assert 'accuracy' in metrics
        assert 'auc' in metrics


class TestRunAllGroupingExperiments:
    """Test helper to run all grouping experiments."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data split."""
        embeddings = torch.randn(100, 64)
        fine_labels = torch.randint(0, 8, (100,))
        return {
            'train_embeddings': embeddings[:60],
            'train_labels': fine_labels[:60],
            'val_embeddings': embeddings[60:80],
            'val_labels': fine_labels[60:80],
            'test_embeddings': embeddings[80:],
            'test_labels': fine_labels[80:],
        }

    def test_run_all_experiments(self, sample_data):
        """Should run binary, ternary, and 8-class experiments."""
        from experiments.class_imbalance.class_grouping import run_all_grouping_experiments

        results = run_all_grouping_experiments(
            **sample_data,
            epochs=2,
        )

        assert 'binary' in results
        assert 'ternary' in results
        assert 'full' in results

        for grouping, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'auc' in metrics
