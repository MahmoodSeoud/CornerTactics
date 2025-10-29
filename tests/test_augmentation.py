#!/usr/bin/env python3
"""
Unit tests for D2 augmentation module.

Tests the implementation of D2 (dihedral group) symmetry transformations
for corner kick graphs following TacticAI methodology.
"""

import pytest
import numpy as np
import torch
from src.data.augmentation import D2Augmentation


# StatsBomb pitch dimensions
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


class TestD2Augmentation:
    """Test suite for D2 augmentation transformations."""

    @pytest.fixture
    def augmentation(self):
        """Create D2Augmentation instance."""
        return D2Augmentation(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)

    @pytest.fixture
    def sample_features(self):
        """
        Create sample 14-dimensional feature tensor for testing.

        Features: [x, y, dist_goal, dist_ball, vx, vy, vel_mag, vel_angle,
                   angle_goal, angle_ball, team, penalty_box, num_nearby, density]
        """
        # Single player at (100, 50) with velocity (2, 1)
        return torch.tensor([
            [100.0, 50.0, 22.36, 15.0, 2.0, 1.0, 2.236, 0.464,
             0.927, 0.644, 1.0, 1.0, 3.0, 0.85]
        ], dtype=torch.float32)

    @pytest.fixture
    def sample_edge_index(self):
        """Create sample edge index for a simple 3-node graph."""
        # Graph with 3 nodes: 0->1, 1->2, 2->0
        return torch.tensor([
            [0, 1, 2],
            [1, 2, 0]
        ], dtype=torch.long)

    def test_identity_transform(self, augmentation, sample_features):
        """Test that identity transformation returns unchanged features."""
        transformed = augmentation.apply_transform(sample_features, 'identity')

        assert torch.allclose(transformed, sample_features, atol=1e-6)

    def test_h_flip_position(self, augmentation, sample_features):
        """Test horizontal flip of x position (x -> 120 - x)."""
        transformed = augmentation.apply_transform(sample_features, 'h_flip')

        # Original x=100 should become x=20 (120-100)
        assert torch.isclose(transformed[0, 0], torch.tensor(20.0), atol=1e-6)
        # Y should be unchanged
        assert torch.isclose(transformed[0, 1], sample_features[0, 1], atol=1e-6)

    def test_h_flip_velocity(self, augmentation, sample_features):
        """Test horizontal flip of vx velocity (vx -> -vx)."""
        transformed = augmentation.apply_transform(sample_features, 'h_flip')

        # vx should be negated: 2.0 -> -2.0
        assert torch.isclose(transformed[0, 4], torch.tensor(-2.0), atol=1e-6)
        # vy should be unchanged
        assert torch.isclose(transformed[0, 5], sample_features[0, 5], atol=1e-6)

    def test_v_flip_position(self, augmentation, sample_features):
        """Test vertical flip of y position (y -> 80 - y)."""
        transformed = augmentation.apply_transform(sample_features, 'v_flip')

        # X should be unchanged
        assert torch.isclose(transformed[0, 0], sample_features[0, 0], atol=1e-6)
        # Original y=50 should become y=30 (80-50)
        assert torch.isclose(transformed[0, 1], torch.tensor(30.0), atol=1e-6)

    def test_v_flip_velocity(self, augmentation, sample_features):
        """Test vertical flip of vy velocity (vy -> -vy)."""
        transformed = augmentation.apply_transform(sample_features, 'v_flip')

        # vx should be unchanged
        assert torch.isclose(transformed[0, 4], sample_features[0, 4], atol=1e-6)
        # vy should be negated: 1.0 -> -1.0
        assert torch.isclose(transformed[0, 5], torch.tensor(-1.0), atol=1e-6)

    def test_both_flip_position(self, augmentation, sample_features):
        """Test both flips (180 degree rotation)."""
        transformed = augmentation.apply_transform(sample_features, 'both_flip')

        # x=100 -> x=20, y=50 -> y=30
        assert torch.isclose(transformed[0, 0], torch.tensor(20.0), atol=1e-6)
        assert torch.isclose(transformed[0, 1], torch.tensor(30.0), atol=1e-6)

    def test_both_flip_velocity(self, augmentation, sample_features):
        """Test both flips on velocity (both components negated)."""
        transformed = augmentation.apply_transform(sample_features, 'both_flip')

        # vx: 2.0 -> -2.0, vy: 1.0 -> -1.0
        assert torch.isclose(transformed[0, 4], torch.tensor(-2.0), atol=1e-6)
        assert torch.isclose(transformed[0, 5], torch.tensor(-1.0), atol=1e-6)

    def test_h_flip_twice_is_identity(self, augmentation, sample_features):
        """Test that applying h_flip twice returns to original state."""
        once = augmentation.apply_transform(sample_features.clone(), 'h_flip')
        twice = augmentation.apply_transform(once, 'h_flip')

        assert torch.allclose(twice, sample_features, atol=1e-5)

    def test_v_flip_twice_is_identity(self, augmentation, sample_features):
        """Test that applying v_flip twice returns to original state."""
        once = augmentation.apply_transform(sample_features.clone(), 'v_flip')
        twice = augmentation.apply_transform(once, 'v_flip')

        assert torch.allclose(twice, sample_features, atol=1e-5)

    def test_both_flip_twice_is_identity(self, augmentation, sample_features):
        """Test that applying both_flip twice returns to original state."""
        once = augmentation.apply_transform(sample_features.clone(), 'both_flip')
        twice = augmentation.apply_transform(once, 'both_flip')

        assert torch.allclose(twice, sample_features, atol=1e-5)

    def test_edge_structure_unchanged(self, augmentation, sample_features, sample_edge_index):
        """Test that edge structure is preserved across transformations."""
        original_edges = sample_edge_index.clone()

        # Apply each transformation
        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            # Transform features (edges should be unchanged)
            _ = augmentation.apply_transform(sample_features, transform_type)
            # Edge index should remain unchanged
            assert torch.equal(sample_edge_index, original_edges)

    def test_feature_ranges_preserved(self, augmentation, sample_features):
        """Test that x and y remain in valid pitch bounds after transformation."""
        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            transformed = augmentation.apply_transform(sample_features, transform_type)

            # X should be in [0, 120]
            assert 0 <= transformed[0, 0] <= PITCH_LENGTH
            # Y should be in [0, 80]
            assert 0 <= transformed[0, 1] <= PITCH_WIDTH

    def test_team_flag_unchanged(self, augmentation, sample_features):
        """Test that team flag is not affected by geometric transformations."""
        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            transformed = augmentation.apply_transform(sample_features, transform_type)

            # Team flag (index 10) should remain unchanged
            assert torch.isclose(transformed[0, 10], sample_features[0, 10], atol=1e-6)

    def test_get_all_views(self, augmentation, sample_features, sample_edge_index):
        """Test generation of all 4 D2 views."""
        views = augmentation.get_all_views(sample_features, sample_edge_index)

        # Should return list of 4 views
        assert len(views) == 4

        # Each view should have same shape as input
        for view_x, view_edge in views:
            assert view_x.shape == sample_features.shape
            assert view_edge.shape == sample_edge_index.shape

    def test_get_all_views_order(self, augmentation, sample_features, sample_edge_index):
        """Test that get_all_views returns views in correct order."""
        views = augmentation.get_all_views(sample_features, sample_edge_index)

        # Extract just the feature tensors
        view_tensors = [view[0] for view in views]

        # View 0: identity
        assert torch.allclose(view_tensors[0], sample_features, atol=1e-6)

        # View 1: h_flip
        h_flip = augmentation.apply_transform(sample_features.clone(), 'h_flip')
        assert torch.allclose(view_tensors[1], h_flip, atol=1e-6)

        # View 2: v_flip
        v_flip = augmentation.apply_transform(sample_features.clone(), 'v_flip')
        assert torch.allclose(view_tensors[2], v_flip, atol=1e-6)

        # View 3: both_flip
        both_flip = augmentation.apply_transform(sample_features.clone(), 'both_flip')
        assert torch.allclose(view_tensors[3], both_flip, atol=1e-6)

    def test_batch_transformation(self, augmentation):
        """Test transformation on batched features (multiple players)."""
        # Create batch of 3 players
        batch_features = torch.tensor([
            [100.0, 50.0, 22.36, 15.0, 2.0, 1.0, 2.236, 0.464,
             0.927, 0.644, 1.0, 1.0, 3.0, 0.85],
            [110.0, 60.0, 20.0, 10.0, -1.0, 0.5, 1.118, 2.678,
             1.107, 0.785, 0.0, 1.0, 5.0, 1.2],
            [90.0, 40.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0,
             0.643, 0.464, 1.0, 1.0, 2.0, 0.6]
        ], dtype=torch.float32)

        transformed = augmentation.apply_transform(batch_features, 'h_flip')

        # Check that all 3 players were transformed
        assert transformed.shape == batch_features.shape

        # Check first player x coordinate: 100 -> 20
        assert torch.isclose(transformed[0, 0], torch.tensor(20.0), atol=1e-6)
        # Check second player x coordinate: 110 -> 10
        assert torch.isclose(transformed[1, 0], torch.tensor(10.0), atol=1e-6)

    def test_invalid_transform_type(self, augmentation, sample_features):
        """Test that invalid transform type raises appropriate error."""
        with pytest.raises(ValueError):
            augmentation.apply_transform(sample_features, 'invalid_transform')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
