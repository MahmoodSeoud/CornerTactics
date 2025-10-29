#!/usr/bin/env python3
"""
D2 Augmentation for TacticAI-Style Corner Prediction

Implements Day 8-9: D2 Augmentation Implementation
Provides dihedral group D2 transformations for spatial data augmentation.

D2 symmetry group has 4 elements:
1. Identity (no transformation)
2. Horizontal flip (mirror across vertical axis)
3. Vertical flip (mirror across horizontal axis)
4. Both flips (180-degree rotation)

StatsBomb pitch coordinates: 120x80 units
- X: 0 (defensive end) to 120 (attacking end)
- Y: 0 (bottom) to 80 (top)

Transformations:
- H-flip: x' = 120 - x, vx' = -vx
- V-flip: y' = 80 - y, vy' = -vy
- Both: Apply both transformations

Author: mseo
Date: October 2024
"""

import torch
from typing import Tuple, List


class D2Augmentation:
    """
    D2 symmetry augmentation for corner kick graphs.

    Implements geometric transformations that preserve the structure
    of corner kick scenarios while providing data augmentation.

    Transformations:
    - identity: No change
    - h_flip: Horizontal flip (mirror left-right)
    - v_flip: Vertical flip (mirror top-bottom)
    - both_flip: Both flips (equivalent to 180-degree rotation)
    """

    def __init__(self, pitch_width: float = 120.0, pitch_height: float = 80.0):
        """
        Initialize D2 augmentation with pitch dimensions.

        Args:
            pitch_width: Width of pitch (default 120.0 for StatsBomb)
            pitch_height: Height of pitch (default 80.0 for StatsBomb)
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

    def horizontal_flip(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply horizontal flip transformation.

        Mirrors positions across vertical axis (x = 60).
        Negates x-component of velocity.

        Args:
            positions: Player positions [num_players, 2] (x, y)
            velocities: Player velocities [num_players, 2] (vx, vy)

        Returns:
            Tuple of (flipped_positions, flipped_velocities)
        """
        flipped_positions = positions.clone()
        flipped_velocities = velocities.clone()

        # Flip x coordinate: x' = 120 - x
        flipped_positions[:, 0] = self.pitch_width - positions[:, 0]

        # Flip x velocity: vx' = -vx
        flipped_velocities[:, 0] = -velocities[:, 0]

        return flipped_positions, flipped_velocities

    def vertical_flip(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply vertical flip transformation.

        Mirrors positions across horizontal axis (y = 40).
        Negates y-component of velocity.

        Args:
            positions: Player positions [num_players, 2] (x, y)
            velocities: Player velocities [num_players, 2] (vx, vy)

        Returns:
            Tuple of (flipped_positions, flipped_velocities)
        """
        flipped_positions = positions.clone()
        flipped_velocities = velocities.clone()

        # Flip y coordinate: y' = 80 - y
        flipped_positions[:, 1] = self.pitch_height - positions[:, 1]

        # Flip y velocity: vy' = -vy
        flipped_velocities[:, 1] = -velocities[:, 1]

        return flipped_positions, flipped_velocities

    def both_flip(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply both horizontal and vertical flips.

        Equivalent to 180-degree rotation around pitch center.

        Args:
            positions: Player positions [num_players, 2] (x, y)
            velocities: Player velocities [num_players, 2] (vx, vy)

        Returns:
            Tuple of (flipped_positions, flipped_velocities)
        """
        # Apply h-flip
        pos, vel = self.horizontal_flip(positions, velocities)

        # Then apply v-flip
        pos, vel = self.vertical_flip(pos, vel)

        return pos, vel

    def apply_transform(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.Tensor,
        transform_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply specified D2 transformation.

        Args:
            positions: Player positions [num_players, 2]
            velocities: Player velocities [num_players, 2]
            edge_index: Graph edge indices [2, num_edges]
            transform_type: One of ['identity', 'h_flip', 'v_flip', 'both_flip']

        Returns:
            Tuple of (transformed_positions, transformed_velocities, edge_index)

        Note:
            Edge index is unchanged by spatial transformations
        """
        if transform_type == 'identity':
            return positions.clone(), velocities.clone(), edge_index.clone()
        elif transform_type == 'h_flip':
            pos, vel = self.horizontal_flip(positions, velocities)
            return pos, vel, edge_index.clone()
        elif transform_type == 'v_flip':
            pos, vel = self.vertical_flip(positions, velocities)
            return pos, vel, edge_index.clone()
        elif transform_type == 'both_flip':
            pos, vel = self.both_flip(positions, velocities)
            return pos, vel, edge_index.clone()
        else:
            raise ValueError(
                f"Unknown transform_type: {transform_type}. "
                f"Must be one of ['identity', 'h_flip', 'v_flip', 'both_flip']"
            )

    def get_all_views(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate all 4 D2 views of a corner kick.

        Args:
            positions: Player positions [num_players, 2]
            velocities: Player velocities [num_players, 2]
            edge_index: Graph edge indices [2, num_edges]

        Returns:
            List of 4 tuples: [(pos, vel, edges), ...]
            Order: identity, h_flip, v_flip, both_flip
        """
        views = []

        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            view = self.apply_transform(
                positions, velocities, edge_index, transform_type
            )
            views.append(view)

        return views


if __name__ == "__main__":
    # Quick test
    print("="*60)
    print("TESTING D2 AUGMENTATION")
    print("="*60)

    augmenter = D2Augmentation()

    # Sample data
    positions = torch.tensor([
        [30.0, 20.0],
        [90.0, 60.0],
        [60.0, 40.0],
    ])

    velocities = torch.tensor([
        [2.0, 1.0],
        [-1.5, 0.5],
        [0.0, -1.0],
    ])

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    print("\nOriginal positions:")
    print(positions)

    print("\nTesting all 4 D2 transforms:")
    views = augmenter.get_all_views(positions, velocities, edge_index)

    for i, (pos, vel, edges) in enumerate(views):
        transform_names = ['identity', 'h_flip', 'v_flip', 'both_flip']
        print(f"\n{i+1}. {transform_names[i]}:")
        print(f"   Positions: {pos.tolist()}")

    print("\n" + "="*60)
    print("✅ D2 AUGMENTATION IMPLEMENTED!")
    print("="*60)
