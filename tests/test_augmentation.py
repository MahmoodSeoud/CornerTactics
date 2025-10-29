#!/usr/bin/env python3
"""
Unit Tests for D2 Augmentation (Days 8-9)

Tests the D2Augmentation class for corner kick spatial transformations.
Validates that:
1. H-flip twice returns identity
2. V-flip twice returns identity
3. Edge structures remain unchanged
4. All 4 D2 transforms produce valid coordinates
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.augmentation import D2Augmentation


class TestD2Augmentation(unittest.TestCase):
    """Test suite for D2 Augmentation transformations."""

    def setUp(self):
        """Set up test fixtures."""
        self.augmenter = D2Augmentation()

        # Create sample player positions (StatsBomb pitch: 120x80)
        # 4 players at different locations
        self.sample_positions = torch.tensor([
            [30.0, 20.0],  # Left side, bottom
            [90.0, 60.0],  # Right side, top
            [60.0, 40.0],  # Center
            [105.0, 10.0], # Far right, bottom
        ])

        # Sample velocities (vx, vy)
        self.sample_velocities = torch.tensor([
            [2.0, 1.0],
            [-1.5, 0.5],
            [0.0, -1.0],
            [1.0, 0.0],
        ])

        # Sample edge index (fully connected for simplicity)
        self.edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
        ])

    def test_horizontal_flip_twice_is_identity(self):
        """Test that applying h-flip twice returns original positions."""
        # Apply h-flip twice
        flipped_once = self.augmenter.horizontal_flip(
            self.sample_positions.clone(),
            self.sample_velocities.clone()
        )
        flipped_twice = self.augmenter.horizontal_flip(
            flipped_once[0].clone(),
            flipped_once[1].clone()
        )

        # Check positions are restored (within floating point tolerance)
        torch.testing.assert_close(
            flipped_twice[0], self.sample_positions,
            rtol=1e-5, atol=1e-5
        )

        # Check velocities are restored
        torch.testing.assert_close(
            flipped_twice[1], self.sample_velocities,
            rtol=1e-5, atol=1e-5
        )

    def test_vertical_flip_twice_is_identity(self):
        """Test that applying v-flip twice returns original positions."""
        # Apply v-flip twice
        flipped_once = self.augmenter.vertical_flip(
            self.sample_positions.clone(),
            self.sample_velocities.clone()
        )
        flipped_twice = self.augmenter.vertical_flip(
            flipped_once[0].clone(),
            flipped_once[1].clone()
        )

        # Check positions are restored
        torch.testing.assert_close(
            flipped_twice[0], self.sample_positions,
            rtol=1e-5, atol=1e-5
        )

        # Check velocities are restored
        torch.testing.assert_close(
            flipped_twice[1], self.sample_velocities,
            rtol=1e-5, atol=1e-5
        )

    def test_edge_structure_unchanged(self):
        """Test that edge index remains unchanged after transformations."""
        # Apply all 4 transformations
        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            _, _, edge_index_out = self.augmenter.apply_transform(
                self.sample_positions.clone(),
                self.sample_velocities.clone(),
                self.edge_index.clone(),
                transform_type
            )

            # Edge index should be identical
            torch.testing.assert_close(edge_index_out, self.edge_index)

    def test_all_transforms_produce_valid_coordinates(self):
        """Test that all 4 D2 transforms produce valid StatsBomb coordinates."""
        pitch_width = 120.0
        pitch_height = 80.0

        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            positions_out, _, _ = self.augmenter.apply_transform(
                self.sample_positions.clone(),
                self.sample_velocities.clone(),
                self.edge_index.clone(),
                transform_type
            )

            # Check x coordinates are in [0, 120]
            self.assertTrue(torch.all(positions_out[:, 0] >= 0))
            self.assertTrue(torch.all(positions_out[:, 0] <= pitch_width))

            # Check y coordinates are in [0, 80]
            self.assertTrue(torch.all(positions_out[:, 1] >= 0))
            self.assertTrue(torch.all(positions_out[:, 1] <= pitch_height))

    def test_horizontal_flip_x_coordinate(self):
        """Test that h-flip correctly transforms x coordinates."""
        positions_flipped, _, _ = self.augmenter.apply_transform(
            self.sample_positions.clone(),
            self.sample_velocities.clone(),
            self.edge_index.clone(),
            'h_flip'
        )

        # x' = 120 - x
        expected_x = 120.0 - self.sample_positions[:, 0]
        torch.testing.assert_close(
            positions_flipped[:, 0], expected_x,
            rtol=1e-5, atol=1e-5
        )

        # y should be unchanged
        torch.testing.assert_close(
            positions_flipped[:, 1], self.sample_positions[:, 1],
            rtol=1e-5, atol=1e-5
        )

    def test_vertical_flip_y_coordinate(self):
        """Test that v-flip correctly transforms y coordinates."""
        positions_flipped, _, _ = self.augmenter.apply_transform(
            self.sample_positions.clone(),
            self.sample_velocities.clone(),
            self.edge_index.clone(),
            'v_flip'
        )

        # y' = 80 - y
        expected_y = 80.0 - self.sample_positions[:, 1]
        torch.testing.assert_close(
            positions_flipped[:, 1], expected_y,
            rtol=1e-5, atol=1e-5
        )

        # x should be unchanged
        torch.testing.assert_close(
            positions_flipped[:, 0], self.sample_positions[:, 0],
            rtol=1e-5, atol=1e-5
        )

    def test_get_all_views(self):
        """Test that get_all_views generates 4 D2 views."""
        views = self.augmenter.get_all_views(
            self.sample_positions,
            self.sample_velocities,
            self.edge_index
        )

        # Should have 4 views
        self.assertEqual(len(views), 4)

        # Each view should have positions, velocities, edge_index
        for view in views:
            self.assertEqual(len(view), 3)
            positions, velocities, edge_index = view

            # Check shapes
            self.assertEqual(positions.shape, self.sample_positions.shape)
            self.assertEqual(velocities.shape, self.sample_velocities.shape)
            self.assertEqual(edge_index.shape, self.edge_index.shape)

    def test_velocity_flipping(self):
        """Test that velocities are correctly flipped."""
        # H-flip: vx should be negated, vy unchanged
        _, vel_h, _ = self.augmenter.apply_transform(
            self.sample_positions.clone(),
            self.sample_velocities.clone(),
            self.edge_index.clone(),
            'h_flip'
        )
        expected_vx = -self.sample_velocities[:, 0]
        torch.testing.assert_close(vel_h[:, 0], expected_vx, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vel_h[:, 1], self.sample_velocities[:, 1], rtol=1e-5, atol=1e-5)

        # V-flip: vy should be negated, vx unchanged
        _, vel_v, _ = self.augmenter.apply_transform(
            self.sample_positions.clone(),
            self.sample_velocities.clone(),
            self.edge_index.clone(),
            'v_flip'
        )
        expected_vy = -self.sample_velocities[:, 1]
        torch.testing.assert_close(vel_v[:, 0], self.sample_velocities[:, 0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vel_v[:, 1], expected_vy, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
