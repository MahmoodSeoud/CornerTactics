#!/usr/bin/env python3
"""
Feature Engineering Module for Corner Kick GNN

Implements Phase 2.1: Node Feature Engineering
Creates 12-dimensional feature vectors per player:
- Spatial: x, y, distance_to_goal, distance_to_ball_target
- Kinematic: vx, vy, velocity_magnitude, velocity_angle
- Contextual: angle_to_goal, angle_to_ball, team_flag, in_penalty_box
- Density: num_players_within_5m, local_density_score (added as 13th and 14th features)

Based on Bekkers & Sahasrabudhe (2024) GNN methodology.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# StatsBomb pitch dimensions (120 x 80 yards)
PITCH_LENGTH = 120.0  # x-axis
PITCH_WIDTH = 80.0     # y-axis

# Goal coordinates (center of goal at x=120)
GOAL_CENTER = (120.0, 40.0)
GOAL_WIDTH = 8.0  # StatsBomb goal width

# Penalty box dimensions (StatsBomb)
PENALTY_BOX_LENGTH = 18.0  # From goal line
PENALTY_BOX_X_MIN = 102.0  # 120 - 18
PENALTY_BOX_Y_MIN = 18.0   # (80 - 44) / 2
PENALTY_BOX_Y_MAX = 62.0   # 18 + 44

# Density calculation parameters
DENSITY_RADIUS_5M = 5.0  # meters (approximate in pitch units)
DENSITY_KERNEL_SIGMA = 3.0


@dataclass
class PlayerFeatures:
    """
    Complete feature vector for a single player node in the graph.

    Attributes:
        player_id: Unique identifier for player
        team: 'attacking' or 'defending'

        # Spatial features (4)
        x: X coordinate (0-120)
        y: Y coordinate (0-80)
        distance_to_goal: Euclidean distance to goal center
        distance_to_ball_target: Distance to ball landing zone

        # Kinematic features (4)
        vx: Velocity in x direction (only for SkillCorner tracking)
        vy: Velocity in y direction
        velocity_magnitude: Speed (sqrt(vx^2 + vy^2))
        velocity_angle: Direction of movement (radians)

        # Contextual features (4)
        angle_to_goal: Angle from player to goal (radians)
        angle_to_ball: Angle from player to ball landing zone
        team_flag: 1.0 for attacking, 0.0 for defending
        in_penalty_box: 1.0 if in penalty box, 0.0 otherwise

        # Density features (2) - bonus features
        num_players_within_5m: Count of nearby players
        local_density_score: Gaussian kernel density score
    """
    player_id: str
    team: str

    # Spatial (4)
    x: float
    y: float
    distance_to_goal: float
    distance_to_ball_target: float

    # Kinematic (4)
    vx: float = 0.0
    vy: float = 0.0
    velocity_magnitude: float = 0.0
    velocity_angle: float = 0.0

    # Contextual (4)
    angle_to_goal: float = 0.0
    angle_to_ball: float = 0.0
    team_flag: float = 1.0
    in_penalty_box: float = 0.0

    # Density (2)
    num_players_within_5m: int = 0
    local_density_score: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to 14-dimensional feature vector."""
        return np.array([
            self.x,
            self.y,
            self.distance_to_goal,
            self.distance_to_ball_target,
            self.vx,
            self.vy,
            self.velocity_magnitude,
            self.velocity_angle,
            self.angle_to_goal,
            self.angle_to_ball,
            self.team_flag,
            self.in_penalty_box,
            self.num_players_within_5m,
            self.local_density_score
        ])


class FeatureEngineer:
    """
    Main class for engineering node features for corner kick GNN.

    Handles both StatsBomb 360 freeze frames and SkillCorner continuous tracking.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH):
        """
        Initialize feature engineer.

        Args:
            pitch_length: Pitch length in StatsBomb units (default 120)
            pitch_width: Pitch width in StatsBomb units (default 80)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_center = GOAL_CENTER

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1: (x1, y1) coordinates
            pos2: (x2, y2) coordinates

        Returns:
            Euclidean distance
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def calculate_angle(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
        """
        Calculate angle from one position to another.

        Args:
            from_pos: Starting position (x1, y1)
            to_pos: Target position (x2, y2)

        Returns:
            Angle in radians [-π, π]
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return np.arctan2(dy, dx)

    def is_in_penalty_box(self, x: float, y: float) -> bool:
        """
        Check if position is inside the penalty box.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if inside penalty box
        """
        return (x >= PENALTY_BOX_X_MIN and
                y >= PENALTY_BOX_Y_MIN and
                y <= PENALTY_BOX_Y_MAX)

    def calculate_velocity_features(
        self,
        current_pos: Tuple[float, float],
        previous_pos: Optional[Tuple[float, float]] = None,
        fps: float = 10.0
    ) -> Tuple[float, float, float, float]:
        """
        Calculate velocity features from position tracking.

        Args:
            current_pos: Current (x, y) position
            previous_pos: Previous (x, y) position (if available)
            fps: Frames per second (SkillCorner = 10 fps)

        Returns:
            Tuple of (vx, vy, velocity_magnitude, velocity_angle)
        """
        if previous_pos is None:
            # No velocity data available (static freeze frame)
            return 0.0, 0.0, 0.0, 0.0

        # Calculate velocity components
        dt = 1.0 / fps
        vx = (current_pos[0] - previous_pos[0]) / dt
        vy = (current_pos[1] - previous_pos[1]) / dt

        # Calculate magnitude and angle
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        velocity_angle = np.arctan2(vy, vx)

        return vx, vy, velocity_magnitude, velocity_angle

    def calculate_density_features(
        self,
        player_pos: Tuple[float, float],
        all_positions: List[Tuple[float, float]],
        radius: float = DENSITY_RADIUS_5M
    ) -> Tuple[int, float]:
        """
        Calculate local density features around a player.

        Args:
            player_pos: (x, y) position of the player
            all_positions: List of all player positions in the scene
            radius: Radius for counting nearby players

        Returns:
            Tuple of (num_players_within_radius, density_score)
        """
        distances = [self.calculate_distance(player_pos, other_pos)
                    for other_pos in all_positions
                    if other_pos != player_pos]

        # Count players within radius
        num_nearby = sum(1 for d in distances if d <= radius)

        # Gaussian kernel density score
        density_score = sum(np.exp(-(d**2) / (2 * DENSITY_KERNEL_SIGMA**2))
                          for d in distances)

        return num_nearby, density_score

    def extract_player_features(
        self,
        player_pos: Tuple[float, float],
        player_id: str,
        team: str,
        ball_target: Tuple[float, float],
        all_positions: List[Tuple[float, float]],
        previous_pos: Optional[Tuple[float, float]] = None,
        fps: float = 10.0
    ) -> PlayerFeatures:
        """
        Extract complete feature vector for a single player.

        Args:
            player_pos: Current (x, y) position
            player_id: Player identifier
            team: 'attacking' or 'defending'
            ball_target: Ball landing position (end_x, end_y)
            all_positions: All player positions for density calculation
            previous_pos: Previous position for velocity (optional)
            fps: Tracking frame rate (default 10 fps)

        Returns:
            PlayerFeatures object with all 14 features
        """
        x, y = player_pos

        # Spatial features
        distance_to_goal = self.calculate_distance(player_pos, self.goal_center)
        distance_to_ball_target = self.calculate_distance(player_pos, ball_target)

        # Kinematic features
        vx, vy, velocity_magnitude, velocity_angle = self.calculate_velocity_features(
            player_pos, previous_pos, fps
        )

        # Contextual features
        angle_to_goal = self.calculate_angle(player_pos, self.goal_center)
        angle_to_ball = self.calculate_angle(player_pos, ball_target)
        team_flag = 1.0 if team == 'attacking' else 0.0
        in_penalty_box = 1.0 if self.is_in_penalty_box(x, y) else 0.0

        # Density features
        num_nearby, density_score = self.calculate_density_features(
            player_pos, all_positions
        )

        return PlayerFeatures(
            player_id=player_id,
            team=team,
            x=x,
            y=y,
            distance_to_goal=distance_to_goal,
            distance_to_ball_target=distance_to_ball_target,
            vx=vx,
            vy=vy,
            velocity_magnitude=velocity_magnitude,
            velocity_angle=velocity_angle,
            angle_to_goal=angle_to_goal,
            angle_to_ball=angle_to_ball,
            team_flag=team_flag,
            in_penalty_box=in_penalty_box,
            num_players_within_5m=num_nearby,
            local_density_score=density_score
        )

    def extract_features_from_statsbomb_corner(
        self,
        corner_row: pd.Series
    ) -> List[PlayerFeatures]:
        """
        Extract features from a StatsBomb corner kick row.

        StatsBomb data structure:
        - attacking_positions: JSON string with list of [x, y] positions
        - defending_positions: JSON string with list of [x, y] positions
        - end_x, end_y: Ball landing target

        Args:
            corner_row: Row from StatsBomb corners CSV

        Returns:
            List of PlayerFeatures for all visible players
        """
        # Parse player positions from JSON
        attacking_positions = json.loads(corner_row['attacking_positions'])
        defending_positions = json.loads(corner_row['defending_positions'])

        # Ball target location
        ball_target = (corner_row['end_x'], corner_row['end_y'])

        # Collect all positions for density calculation
        all_positions = attacking_positions + defending_positions

        player_features = []

        # Process attacking players
        for i, pos in enumerate(attacking_positions):
            player_id = f"{corner_row['corner_id']}_att_{i}"
            features = self.extract_player_features(
                player_pos=tuple(pos),
                player_id=player_id,
                team='attacking',
                ball_target=ball_target,
                all_positions=all_positions,
                previous_pos=None  # No velocity for StatsBomb
            )
            player_features.append(features)

        # Process defending players
        for i, pos in enumerate(defending_positions):
            player_id = f"{corner_row['corner_id']}_def_{i}"
            features = self.extract_player_features(
                player_pos=tuple(pos),
                player_id=player_id,
                team='defending',
                ball_target=ball_target,
                all_positions=all_positions,
                previous_pos=None  # No velocity for StatsBomb
            )
            player_features.append(features)

        return player_features

    def extract_features_from_skillcorner_corner(
        self,
        corner_row: pd.Series,
        tracking_data: Optional[pd.DataFrame] = None
    ) -> List[PlayerFeatures]:
        """
        Extract features from a SkillCorner corner kick with tracking data.

        SkillCorner data structure:
        - tracking_data: DataFrame with columns [frame, player_id, team_id, x, y]
        - frame_start: Frame number at corner kick moment
        - x_end, y_end: Ball landing target

        Args:
            corner_row: Row from SkillCorner corners CSV
            tracking_data: Tracking DataFrame (10 fps) for the match

        Returns:
            List of PlayerFeatures for all visible players
        """
        if tracking_data is None or corner_row['has_tracking'] == False:
            # No tracking available, return empty list
            return []

        # Get player positions at corner kick frame
        corner_frame = corner_row['frame_start']
        current_frame_data = tracking_data[tracking_data['frame'] == corner_frame]

        # Get previous frame for velocity calculation (10 fps = 0.1s difference)
        previous_frame_data = tracking_data[tracking_data['frame'] == corner_frame - 1]

        # Ball target location
        ball_target = (corner_row['x_end'], corner_row['y_end'])

        # Collect all current positions
        all_positions = [(row['x'], row['y']) for _, row in current_frame_data.iterrows()]

        player_features = []

        for _, player_row in current_frame_data.iterrows():
            player_id = player_row['player_id']
            current_pos = (player_row['x'], player_row['y'])

            # Determine team (attacking or defending)
            team = 'attacking' if player_row['team_id'] == corner_row['team_id'] else 'defending'

            # Get previous position for velocity
            prev_player = previous_frame_data[previous_frame_data['player_id'] == player_id]
            previous_pos = None
            if not prev_player.empty:
                previous_pos = (prev_player.iloc[0]['x'], prev_player.iloc[0]['y'])

            features = self.extract_player_features(
                player_pos=current_pos,
                player_id=str(player_id),
                team=team,
                ball_target=ball_target,
                all_positions=all_positions,
                previous_pos=previous_pos,
                fps=10.0
            )
            player_features.append(features)

        return player_features

    def features_to_dataframe(self, features_list: List[PlayerFeatures]) -> pd.DataFrame:
        """
        Convert list of PlayerFeatures to pandas DataFrame.

        Args:
            features_list: List of PlayerFeatures objects

        Returns:
            DataFrame with one row per player
        """
        data = []
        for features in features_list:
            row = {
                'player_id': features.player_id,
                'team': features.team,
                'x': features.x,
                'y': features.y,
                'distance_to_goal': features.distance_to_goal,
                'distance_to_ball_target': features.distance_to_ball_target,
                'vx': features.vx,
                'vy': features.vy,
                'velocity_magnitude': features.velocity_magnitude,
                'velocity_angle': features.velocity_angle,
                'angle_to_goal': features.angle_to_goal,
                'angle_to_ball': features.angle_to_ball,
                'team_flag': features.team_flag,
                'in_penalty_box': features.in_penalty_box,
                'num_players_within_5m': features.num_players_within_5m,
                'local_density_score': features.local_density_score
            }
            data.append(row)

        return pd.DataFrame(data)


def extract_features_from_dataset(
    corners_csv_path: str,
    dataset_type: str = 'statsbomb',
    tracking_dir: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch extract features from a full corner dataset.

    Args:
        corners_csv_path: Path to corners CSV file
        dataset_type: 'statsbomb' or 'skillcorner'
        tracking_dir: Directory containing tracking data (for SkillCorner)
        output_path: Path to save output CSV (optional)

    Returns:
        DataFrame with corner_id and features for all players
    """
    corners_df = pd.read_csv(corners_csv_path)
    engineer = FeatureEngineer()

    all_results = []

    for idx, corner_row in corners_df.iterrows():
        if dataset_type == 'statsbomb':
            features = engineer.extract_features_from_statsbomb_corner(corner_row)
        elif dataset_type == 'skillcorner':
            # Load tracking data for this match if available
            tracking_data = None
            if tracking_dir and corner_row['has_tracking']:
                tracking_file = Path(tracking_dir) / corner_row['tracking_file']
                if tracking_file.exists():
                    # Load SkillCorner tracking (JSONL format)
                    tracking_data = pd.read_json(tracking_file, lines=True)

            features = engineer.extract_features_from_skillcorner_corner(
                corner_row, tracking_data
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Convert to DataFrame and add corner metadata
        if features:
            features_df = engineer.features_to_dataframe(features)
            features_df['corner_id'] = corner_row.get('corner_id', f"{dataset_type}_{idx}")
            features_df['match_id'] = corner_row.get('match_id', '')
            features_df['dataset'] = dataset_type
            all_results.append(features_df)

    # Concatenate all results
    result_df = pd.concat(all_results, ignore_index=True)

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved features to {output_path}")

    return result_df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module for Corner Kick GNN")
    print("=" * 60)
    print(f"Pitch dimensions: {PITCH_LENGTH} x {PITCH_WIDTH}")
    print(f"Goal center: {GOAL_CENTER}")
    print(f"Penalty box: x >= {PENALTY_BOX_X_MIN}, y in [{PENALTY_BOX_Y_MIN}, {PENALTY_BOX_Y_MAX}]")
    print("\nFeature dimensions per player: 14")
    print("  - Spatial: x, y, distance_to_goal, distance_to_ball_target (4)")
    print("  - Kinematic: vx, vy, velocity_magnitude, velocity_angle (4)")
    print("  - Contextual: angle_to_goal, angle_to_ball, team_flag, in_penalty_box (4)")
    print("  - Density: num_players_within_5m, local_density_score (2)")
