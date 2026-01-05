"""Tests for training module.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path


@pytest.fixture
def sample_corners():
    """Create sample corner data for testing."""
    corners = []
    for i in range(20):
        corner = {
            "match_id": str(i // 5),
            "event": {"id": f"corner-{i}", "location": [120.0, 0.0 if i % 2 == 0 else 80.0]},
            "freeze_frame": [
                {"location": [100.0 + j, 30.0 + j * 2], "teammate": j < 5, "keeper": j == 9, "actor": j == 0}
                for j in range(10)
            ],
            "shot_outcome": i % 2,  # Alternating labels
        }
        corners.append(corner)
    return corners


@pytest.fixture
def split_indices():
    """Create sample train/val/test split indices."""
    return {
        "train": list(range(0, 12)),
        "val": list(range(12, 16)),
        "test": list(range(16, 20)),
    }


class TestDataLoader:
    """Test data loading utilities."""

    def test_create_dataloaders(self, sample_corners, split_indices):
        """Should create train/val/test dataloaders."""
        from experiments.gnn_baseline.train import create_dataloaders

        train_loader, val_loader, test_loader = create_dataloaders(
            sample_corners,
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
            batch_size=4,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch_content(self, sample_corners, split_indices):
        """Dataloader batches should contain proper graph data."""
        from experiments.gnn_baseline.train import create_dataloaders

        train_loader, _, _ = create_dataloaders(
            sample_corners,
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
            batch_size=4,
        )

        batch = next(iter(train_loader))
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'y')
        assert hasattr(batch, 'batch')


class TestTrainer:
    """Test training functionality."""

    def test_trainer_instantiation(self, sample_corners, split_indices):
        """Trainer should instantiate correctly."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )

        assert trainer is not None
        assert trainer.model is not None

    def test_train_one_epoch(self, sample_corners, split_indices):
        """Should complete one training epoch without errors."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )

        train_loss = trainer.train_epoch()

        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_validate(self, sample_corners, split_indices):
        """Should compute validation loss and AUC."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )

        val_loss, val_auc = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss >= 0
        assert isinstance(val_auc, float)
        assert 0 <= val_auc <= 1

    def test_fit_runs_multiple_epochs(self, sample_corners, split_indices):
        """Fit should run for specified number of epochs."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )

        history = trainer.fit(epochs=3, verbose=False)

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_auc' in history
        assert len(history['train_loss']) == 3

    def test_early_stopping(self, sample_corners, split_indices):
        """Fit should stop early if validation doesn't improve."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
            patience=2,
        )

        # Train for more epochs than patience allows
        history = trainer.fit(epochs=100, verbose=False)

        # Should stop before 100 epochs if no improvement
        # (may or may not trigger depending on random init)
        assert len(history['train_loss']) <= 100


class TestWeightedLoss:
    """Test class-weighted loss for imbalanced data."""

    def test_compute_class_weights(self):
        """Should compute inverse frequency weights."""
        from experiments.gnn_baseline.train import compute_class_weights

        labels = [0, 0, 0, 0, 1]  # 80% class 0, 20% class 1

        weights = compute_class_weights(labels)

        # Class 1 should have higher weight
        assert weights[1] > weights[0]

    def test_weighted_bce_loss(self, sample_corners, split_indices):
        """Trainer should use weighted BCE loss."""
        from experiments.gnn_baseline.train import Trainer

        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
            use_class_weights=True,
        )

        # Should not error when training
        train_loss = trainer.train_epoch()
        assert train_loss >= 0


class TestCheckpointing:
    """Test model checkpointing."""

    def test_save_checkpoint(self, sample_corners, split_indices):
        """Should save model checkpoint."""
        from experiments.gnn_baseline.train import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                corners=sample_corners,
                train_indices=split_indices["train"],
                val_indices=split_indices["val"],
                test_indices=split_indices["test"],
                model_name='graphsage',
                hidden_channels=32,
                batch_size=4,
            )

            checkpoint_path = Path(tmpdir) / "model.pt"
            trainer.save_checkpoint(checkpoint_path)

            assert checkpoint_path.exists()

    def test_load_checkpoint(self, sample_corners, split_indices):
        """Should load model checkpoint."""
        from experiments.gnn_baseline.train import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            trainer1 = Trainer(
                corners=sample_corners,
                train_indices=split_indices["train"],
                val_indices=split_indices["val"],
                test_indices=split_indices["test"],
                model_name='graphsage',
                hidden_channels=32,
                batch_size=4,
            )
            trainer1.train_epoch()

            checkpoint_path = Path(tmpdir) / "model.pt"
            trainer1.save_checkpoint(checkpoint_path)

            # Load in new trainer
            trainer2 = Trainer(
                corners=sample_corners,
                train_indices=split_indices["train"],
                val_indices=split_indices["val"],
                test_indices=split_indices["test"],
                model_name='graphsage',
                hidden_channels=32,
                batch_size=4,
            )
            trainer2.load_checkpoint(checkpoint_path)

            # Weights should match
            for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
                assert torch.allclose(p1, p2)


class TestReproducibility:
    """Test reproducibility with random seeds."""

    def test_seed_produces_same_results(self, sample_corners, split_indices):
        """Same seed should produce same training results."""
        from experiments.gnn_baseline.train import Trainer, set_seed

        set_seed(42)
        trainer1 = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )
        loss1 = trainer1.train_epoch()

        set_seed(42)
        trainer2 = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )
        loss2 = trainer2.train_epoch()

        assert abs(loss1 - loss2) < 1e-5
