#!/usr/bin/env python3
"""
Tests for TacticAI Baseline Models

Tests Day 5-6: Baseline receiver prediction models
- RandomReceiverBaseline
- XGBoostReceiverBaseline
- MLPReceiverBaseline
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baselines import (
    RandomReceiverBaseline,
    XGBoostReceiverBaseline,
    MLPReceiverBaseline
)


class TestRandomReceiverBaseline:
    """Test suite for RandomReceiverBaseline"""

    def test_random_baseline_predict_shape(self):
        """Test that predict returns correct shape"""
        model = RandomReceiverBaseline(num_players=22)

        # Create dummy batch data (PyG format)
        batch_size = 16
        num_nodes = 22 * batch_size
        node_features = torch.randn(num_nodes, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 22)

        # Predict (returns probabilities)
        probs = model.predict(node_features, batch_tensor)

        # Check shape
        assert probs.shape == (batch_size, 22), \
            f"Expected shape ({batch_size}, 22), got {probs.shape}"

    def test_random_baseline_probabilities_sum_to_one(self):
        """Test that softmax probabilities sum to 1"""
        model = RandomReceiverBaseline(num_players=22)

        batch_size = 8
        num_nodes = 22 * batch_size
        node_features = torch.randn(num_nodes, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 22)

        probs = model.predict(node_features, batch_tensor)

        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6), \
            f"Probabilities should sum to 1, got {prob_sums}"

    def test_random_baseline_expected_accuracy(self):
        """Test random baseline achieves expected accuracy (~4.5% top-1, ~13.6% top-3)"""
        model = RandomReceiverBaseline(num_players=22)

        # Generate large sample for stable estimates
        batch_size = 1000
        num_nodes = 22 * batch_size
        node_features = torch.randn(num_nodes, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 22)

        # Random true labels
        np.random.seed(42)
        true_labels = np.random.randint(0, 22, size=batch_size)

        # Predict (returns probabilities)
        probs = model.predict(node_features, batch_tensor)

        # Compute top-1 accuracy
        top1_preds = probs.argmax(dim=1).numpy()
        top1_acc = (top1_preds == true_labels).mean()

        # Compute top-3 accuracy
        top3_preds = probs.topk(3, dim=1).indices.numpy()
        top3_acc = np.mean([label in preds for label, preds in zip(true_labels, top3_preds)])

        # Check expected ranges (with some tolerance for randomness)
        assert 0.03 < top1_acc < 0.06, \
            f"Expected top-1 accuracy ~4.5%, got {top1_acc*100:.1f}%"
        assert 0.11 < top3_acc < 0.16, \
            f"Expected top-3 accuracy ~13.6%, got {top3_acc*100:.1f}%"


class TestXGBoostReceiverBaseline:
    """Test suite for XGBoostReceiverBaseline"""

    def test_xgboost_feature_extraction_shape(self):
        """Test feature extraction produces correct shape"""
        model = XGBoostReceiverBaseline()

        # Create dummy graph data (22 players, 14 features)
        node_features = torch.randn(22, 14)

        # Extract features
        features = model.extract_features(node_features)

        # Should have 22 player feature vectors with ~15 features each
        assert features.shape[0] == 22, "Should have 22 player feature vectors"
        assert features.shape[1] >= 10, "Should have at least 10 features per player"

    def test_xgboost_spatial_features(self):
        """Test spatial features are computed correctly"""
        model = XGBoostReceiverBaseline()

        # Create specific positions to test
        # Player at (60, 40) - center of penalty area
        node_features = torch.zeros(22, 14)
        node_features[0, 0] = 60.0  # x (StatsBomb coordinates: 0-120)
        node_features[0, 1] = 40.0  # y (StatsBomb coordinates: 0-80)
        node_features[0, 2] = 60.0  # dist_to_goal (goal at 120, 40)
        node_features[0, 10] = 1.0  # team_flag (attacking)

        features = model.extract_features(node_features)

        # Check x, y positions are preserved
        assert abs(features[0, 0] - 60.0) < 0.1, "X position should be preserved"
        assert abs(features[0, 1] - 40.0) < 0.1, "Y position should be preserved"

        # Distance to goal should be preserved from input
        dist_to_goal = features[0, 2]
        assert abs(dist_to_goal - 60.0) < 0.1, f"Distance to goal should be 60.0, got {dist_to_goal}"

    def test_xgboost_train_predict_pyg_format(self):
        """Test XGBoost can train and predict with PyG batch format"""
        model = XGBoostReceiverBaseline()

        # Create training data (100 corners to ensure all classes represented)
        np.random.seed(42)
        train_x_list = []
        train_batch_list = []
        train_labels = []

        # First ensure all 22 classes are represented
        for i in range(22):
            x = torch.randn(22, 14)
            batch_tensor = torch.full((22,), i, dtype=torch.long)
            train_x_list.append(x)
            train_batch_list.append(batch_tensor)
            train_labels.append(i)  # Ensure class i exists

        # Add more random samples
        for i in range(22, 100):
            x = torch.randn(22, 14)
            batch_tensor = torch.full((22,), i, dtype=torch.long)
            train_x_list.append(x)
            train_batch_list.append(batch_tensor)
            train_labels.append(np.random.randint(0, 22))

        # Train
        model.train(train_x_list, train_batch_list, train_labels)

        # Predict on test data (single graph)
        test_x = torch.randn(22, 14)
        test_batch = torch.zeros(22, dtype=torch.long)
        probs = model.predict(test_x, test_batch)

        # Check shape
        assert probs.shape == (1, 22), f"Expected shape (1, 22), got {probs.shape}"

        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)


class TestMLPReceiverBaseline:
    """Test suite for MLPReceiverBaseline"""

    def test_mlp_forward_shape(self):
        """Test MLP forward pass returns correct shape"""
        model = MLPReceiverBaseline(
            num_features=14,
            num_players=22
        )

        # Create batch of node features (PyG format)
        batch_size = 16
        num_nodes = 22 * batch_size
        x = torch.randn(num_nodes, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 22)

        # Forward pass
        logits = model(x, batch_tensor)

        # Check shape
        assert logits.shape == (batch_size, 22), \
            f"Expected shape ({batch_size}, 22), got {logits.shape}"

    def test_mlp_predict_probabilities(self):
        """Test MLP predict returns valid probabilities"""
        model = MLPReceiverBaseline(num_features=14, num_players=22)

        batch_size = 8
        x = torch.randn(22 * batch_size, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 22)

        probs = model.predict(x, batch_tensor)

        # Check shape
        assert probs.shape == (batch_size, 22)

        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_mlp_train_step(self):
        """Test MLP training step works"""
        model = MLPReceiverBaseline(num_features=14, num_players=22)

        # Create dummy training data
        x = torch.randn(22 * 32, 14)
        batch_tensor = torch.repeat_interleave(torch.arange(32), 22)
        labels = torch.randint(0, 22, (32,))

        # Compute loss
        logits = model(x, batch_tensor)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Check loss is finite
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
