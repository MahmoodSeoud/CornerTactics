#!/usr/bin/env python3
"""
D2 Augmentation Module for TacticAI Corner Kick GNN

Implements D2 (dihedral group of order 2) symmetry transformations for graph data.
This makes the model equivariant to field reflections, following TacticAI methodology.

D2 Group Transformations:
- Identity: No change
- H-flip: Horizontal reflection (flip along x-axis)
- V-flip: Vertical reflection (flip along y-axis)
- Both-flip: 180° rotation (both reflections)
"""

import torch
from typing import List, Tuple, Literal


TransformType = Literal['identity', 'h_flip', 'v_flip', 'both_flip']


class D2Augmentation:
    """
    D2 symmetry augmentation for corner kick graphs.

    Applies dihedral group transformations while preserving graph structure.
    Updates spatial coordinates and velocity components appropriately.

    Feature vector structure (14 dimensions):
    [0] x: X position (0-120)
    [1] y: Y position (0-80)
    [2] distance_to_goal: Distance to goal center
    [3] distance_to_ball_target: Distance to ball landing zone
    [4] vx: Velocity in x direction
    [5] vy: Velocity in y direction
    [6] velocity_magnitude: Speed
    [7] velocity_angle: Direction of movement (radians)
    [8] angle_to_goal: Angle from player to goal
    [9] angle_to_ball: Angle from player to ball
    [10] team_flag: 1.0 for attacking, 0.0 for defending
    [11] in_penalty_box: 1.0 if in penalty box
    [12] num_players_within_5m: Count of nearby players
    [13] local_density_score: Gaussian kernel density
    """

    def __init__(self, pitch_length: float = 120.0, pitch_width: float = 80.0):
        """
        Initialize D2 augmentation with pitch dimensions.

        Args:
            pitch_length: Pitch length (StatsBomb default: 120)
            pitch_width: Pitch width (StatsBomb default: 80)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def apply_transform(
        self,
        x: torch.Tensor,
        transform_type: TransformType
    ) -> torch.Tensor:
        """
        Apply a D2 transformation to node features.

        Args:
            x: Node features tensor [num_nodes, 14]
            transform_type: Type of transformation ('identity', 'h_flip', 'v_flip', 'both_flip')

        Returns:
            Transformed node features [num_nodes, 14]

        Raises:
            ValueError: If transform_type is invalid
        """
        if transform_type == 'identity':
            return x.clone()

        # Clone input to avoid in-place modifications
        x_transformed = x.clone()

        # Apply horizontal flip
        if transform_type in ['h_flip', 'both_flip']:
            # Flip x coordinate: x -> pitch_length - x
            x_transformed[:, 0] = self.pitch_length - x_transformed[:, 0]

            # Flip x velocity: vx -> -vx
            x_transformed[:, 4] = -x_transformed[:, 4]

            # Transform velocity angle algebraically: when vx flips, angle -> π - angle
            # This preserves perfect involution (double flip = identity)
            mask = x_transformed[:, 6] > 1e-6  # velocity_magnitude > 0
            if mask.any():
                angle = torch.pi - x_transformed[mask, 7]
                # Normalize angle to [-π, π] to handle periodicity
                angle = torch.atan2(torch.sin(angle), torch.cos(angle))
                x_transformed[mask, 7] = angle

        # Apply vertical flip
        if transform_type in ['v_flip', 'both_flip']:
            # Flip y coordinate: y -> pitch_width - y
            x_transformed[:, 1] = self.pitch_width - x_transformed[:, 1]

            # Flip y velocity: vy -> -vy
            x_transformed[:, 5] = -x_transformed[:, 5]

            # Transform velocity angle algebraically: when vy flips, angle -> -angle
            mask = x_transformed[:, 6] > 1e-6  # velocity_magnitude > 0
            if mask.any():
                angle = -x_transformed[mask, 7]
                # Normalize angle to [-π, π] to handle periodicity
                angle = torch.atan2(torch.sin(angle), torch.cos(angle))
                x_transformed[mask, 7] = angle

        # Note: angle_to_ball (index 9) and distance_to_ball_target (index 3) depend on
        # ball position, which also transforms. For now, we keep them as-is since they're
        # relative features. In a full implementation, ball target would also transform.

        # Note: Features that remain unchanged:
        # - team_flag (index 10): Team identity is invariant
        # - in_penalty_box (index 11): Recomputed if needed, but typically robust
        # - num_players_within_5m (index 12): Local density is invariant to global transforms
        # - local_density_score (index 13): Same as above

        if transform_type not in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            raise ValueError(f"Invalid transform_type: {transform_type}")

        return x_transformed

    def get_all_views(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate all 4 D2 views of a graph.

        Args:
            x: Node features [num_nodes, 14]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            List of 4 (features, edge_index) tuples, one for each D2 transformation:
            - View 0: identity
            - View 1: h_flip
            - View 2: v_flip
            - View 3: both_flip

        Note:
            Edge structure is preserved across all transformations (graph topology unchanged).
        """
        views = []

        for transform_type in ['identity', 'h_flip', 'v_flip', 'both_flip']:
            x_transformed = self.apply_transform(x, transform_type)
            # Edge structure remains unchanged
            views.append((x_transformed, edge_index))

        return views

    def apply_random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a random D2 transformation (useful for data augmentation during training).

        Args:
            x: Node features [num_nodes, 14]

        Returns:
            Randomly transformed node features
        """
        import random
        transform_type = random.choice(['identity', 'h_flip', 'v_flip', 'both_flip'])
        return self.apply_transform(x, transform_type)


def test_d2_augmentation():
    """Quick test of D2 augmentation functionality."""
    print("Testing D2 Augmentation")
    print("=" * 60)

    # Create augmentation instance
    aug = D2Augmentation()

    # Create sample feature tensor (1 player at position 100, 50)
    x = torch.tensor([
        [100.0, 50.0, 22.36, 15.0, 2.0, 1.0, 2.236, 0.464,
         0.927, 0.644, 1.0, 1.0, 3.0, 0.85]
    ], dtype=torch.float32)

    print(f"Original position: x={x[0, 0]:.1f}, y={x[0, 1]:.1f}")
    print(f"Original velocity: vx={x[0, 4]:.1f}, vy={x[0, 5]:.1f}")

    # Test identity
    x_id = aug.apply_transform(x, 'identity')
    print(f"\nIdentity: x={x_id[0, 0]:.1f}, y={x_id[0, 1]:.1f}, "
          f"vx={x_id[0, 4]:.1f}, vy={x_id[0, 5]:.1f}")

    # Test h_flip
    x_h = aug.apply_transform(x, 'h_flip')
    print(f"H-flip: x={x_h[0, 0]:.1f}, y={x_h[0, 1]:.1f}, "
          f"vx={x_h[0, 4]:.1f}, vy={x_h[0, 5]:.1f}")

    # Test v_flip
    x_v = aug.apply_transform(x, 'v_flip')
    print(f"V-flip: x={x_v[0, 0]:.1f}, y={x_v[0, 1]:.1f}, "
          f"vx={x_v[0, 4]:.1f}, vy={x_v[0, 5]:.1f}")

    # Test both_flip
    x_b = aug.apply_transform(x, 'both_flip')
    print(f"Both-flip: x={x_b[0, 0]:.1f}, y={x_b[0, 1]:.1f}, "
          f"vx={x_b[0, 4]:.1f}, vy={x_b[0, 5]:.1f}")

    # Test involution (flip twice = identity)
    x_hh = aug.apply_transform(x_h, 'h_flip')
    print(f"\nDouble h-flip check: {torch.allclose(x_hh, x, atol=1e-5)}")

    # Test get_all_views
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    views = aug.get_all_views(x, edge_index)
    print(f"\nGenerated {len(views)} views")
    for i, (view_x, view_edge) in enumerate(views):
        print(f"  View {i}: x={view_x[0, 0]:.1f}, y={view_x[0, 1]:.1f}")

    print("\n✓ D2 augmentation test complete")


if __name__ == "__main__":
    test_d2_augmentation()
