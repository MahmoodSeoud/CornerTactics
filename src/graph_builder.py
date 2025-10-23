#!/usr/bin/env python3
"""
Graph Builder Module for Corner Kick GNN

Implements Phase 2.2: Adjacency Matrix Construction
Converts player node features into graph representations with multiple adjacency strategies.

Based on Bekkers & Sahasrabudhe (2024) "A Graph Neural Network Deep-Dive into Successful Counterattacks"

Adjacency Strategies:
1. Team-based: Connect teammates only
2. Team-with-ball: Connect teammates + ball node (US Soccer Fed approach)
3. Distance-based: Connect players within threshold
4. Delaunay: Spatial triangulation
5. Ball-centric: Connect players near ball trajectory
6. Zone-based: Connect players in same tactical zones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import warnings

# Pitch dimensions (StatsBomb)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Adjacency thresholds
DISTANCE_THRESHOLD = 10.0  # meters (approximate in pitch units)
BALL_ZONE_RADIUS = 15.0     # radius around ball landing zone

# Tactical zones (x, y min/max bounds)
TACTICAL_ZONES = {
    'six_yard_box': (114.0, 120.0, 30.0, 50.0),
    'penalty_area': (102.0, 120.0, 18.0, 62.0),
    'goal_mouth': (118.0, 120.0, 36.0, 44.0),
    'near_post': (114.0, 120.0, 18.0, 40.0),
    'far_post': (114.0, 120.0, 40.0, 62.0),
}

AdjacencyStrategy = Literal['team', 'team_with_ball', 'distance', 'delaunay', 'ball_centric', 'zone']


@dataclass
class EdgeFeatures:
    """
    Edge features for a single connection between two players.

    Attributes:
        distance: Euclidean distance between players (normalized)
        relative_vx: Velocity difference in x direction
        relative_vy: Velocity difference in y direction
        angle_sin: Sine of angle between players
        angle_cos: Cosine of angle between players
    """
    distance: float
    relative_vx: float = 0.0
    relative_vy: float = 0.0
    angle_sin: float = 0.0
    angle_cos: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to 6-dimensional edge feature vector."""
        return np.array([
            self.distance,
            self.relative_vx,
            self.relative_vy,
            np.sqrt(self.relative_vx**2 + self.relative_vy**2),  # relative velocity magnitude
            self.angle_sin,
            self.angle_cos
        ])


@dataclass
class CornerGraph:
    """
    Complete graph representation of a corner kick.

    Attributes:
        corner_id: Unique identifier
        node_features: (N_players, 14) array of node features
        adjacency_matrix: (N_players, N_players) sparse adjacency matrix
        edge_features: (N_edges, 6) array of edge features
        edge_index: (2, N_edges) array of edge connectivity (i, j pairs)
        player_ids: List of player IDs corresponding to nodes
        teams: List of team labels ('attacking'/'defending')
        outcome_label: Corner outcome (goal/shot/clearance/etc.)
    """
    corner_id: str
    node_features: np.ndarray
    adjacency_matrix: csr_matrix
    edge_features: np.ndarray
    edge_index: np.ndarray
    player_ids: List[str]
    teams: List[str]
    outcome_label: Optional[str] = None
    goal_scored: bool = False

    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph."""
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of edges in graph."""
        return self.edge_features.shape[0]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'corner_id': self.corner_id,
            'node_features': self.node_features,
            'adjacency_matrix': self.adjacency_matrix,
            'edge_features': self.edge_features,
            'edge_index': self.edge_index,
            'player_ids': self.player_ids,
            'teams': self.teams,
            'outcome_label': self.outcome_label,
            'goal_scored': self.goal_scored,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }


class GraphBuilder:
    """
    Main class for building graph representations of corner kicks.

    Converts player node features into graphs with various adjacency strategies.
    """

    def __init__(
        self,
        adjacency_strategy: AdjacencyStrategy = 'team',
        distance_threshold: float = DISTANCE_THRESHOLD,
        ball_zone_radius: float = BALL_ZONE_RADIUS
    ):
        """
        Initialize graph builder.

        Args:
            adjacency_strategy: Strategy for connecting nodes
            distance_threshold: Distance threshold for 'distance' strategy
            ball_zone_radius: Radius for 'ball_centric' strategy
        """
        self.adjacency_strategy = adjacency_strategy
        self.distance_threshold = distance_threshold
        self.ball_zone_radius = ball_zone_radius

    def build_graph_from_features(
        self,
        features_df: pd.DataFrame,
        corner_id: str
    ) -> CornerGraph:
        """
        Build graph from pre-extracted node features DataFrame.

        Args:
            features_df: DataFrame with columns [player_id, team, x, y, vx, vy, ...]
            corner_id: Unique identifier for this corner

        Returns:
            CornerGraph with adjacency matrix and edge features
        """
        # Extract node feature matrix (14 dimensions)
        feature_columns = [
            'x', 'y', 'distance_to_goal', 'distance_to_ball_target',
            'vx', 'vy', 'velocity_magnitude', 'velocity_angle',
            'angle_to_goal', 'angle_to_ball', 'team_flag', 'in_penalty_box',
            'num_players_within_5m', 'local_density_score'
        ]
        node_features = features_df[feature_columns].values

        # Extract metadata
        player_ids = features_df['player_id'].tolist()
        teams = features_df['team'].tolist()

        # Build adjacency matrix based on strategy
        adjacency_matrix = self._build_adjacency_matrix(features_df)

        # If team_with_ball strategy, add ball node to features
        if self.adjacency_strategy == 'team_with_ball':
            # Ball node features (14 dimensions, mostly zeros except position)
            ball_x = features_df.iloc[0]['x']
            ball_y = features_df.iloc[0]['y']
            ball_features = np.zeros(14)
            ball_features[0] = ball_x  # x position
            ball_features[1] = ball_y  # y position
            # distance_to_goal, distance_to_ball_target would be calculated but set to 0 for simplicity
            # Ball has no velocity, angle features are 0
            # team_flag = 0 (neutral)

            node_features = np.vstack([node_features, ball_features])
            player_ids.append('ball')
            teams.append('ball')

        # Compute edge features
        edge_index, edge_features = self._compute_edge_features(features_df, adjacency_matrix)

        # Extract outcome label
        outcome_label = features_df['outcome_category'].iloc[0] if 'outcome_category' in features_df.columns else None
        goal_scored = bool(features_df['goal_scored'].iloc[0]) if 'goal_scored' in features_df.columns else False

        return CornerGraph(
            corner_id=corner_id,
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            edge_features=edge_features,
            edge_index=edge_index,
            player_ids=player_ids,
            teams=teams,
            outcome_label=outcome_label,
            goal_scored=goal_scored
        )

    def _build_adjacency_matrix(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build adjacency matrix using selected strategy.

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix (N_players, N_players)
        """
        if self.adjacency_strategy == 'team':
            return self._build_team_based_adjacency(features_df)
        elif self.adjacency_strategy == 'team_with_ball':
            return self._build_team_with_ball_adjacency(features_df)
        elif self.adjacency_strategy == 'distance':
            return self._build_distance_based_adjacency(features_df)
        elif self.adjacency_strategy == 'delaunay':
            return self._build_delaunay_adjacency(features_df)
        elif self.adjacency_strategy == 'ball_centric':
            return self._build_ball_centric_adjacency(features_df)
        elif self.adjacency_strategy == 'zone':
            return self._build_zone_based_adjacency(features_df)
        else:
            raise ValueError(f"Unknown adjacency strategy: {self.adjacency_strategy}")

    def _build_team_based_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build team-based adjacency: Connect teammates only.

        Paper baseline: "Within the adjacency matrix players from the same team
        are connected to each other and every player is connected to the ball node."

        Note: Ball node is implicit in ball-related features (distance_to_ball_target, angle_to_ball)

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix
        """
        n = len(features_df)
        adjacency = np.zeros((n, n), dtype=np.float32)

        # Connect teammates
        teams = features_df['team'].values
        for i in range(n):
            for j in range(i + 1, n):
                if teams[i] == teams[j]:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0  # Undirected graph

        return csr_matrix(adjacency)

    def _build_team_with_ball_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build team-with-ball adjacency: Connect teammates + ball node to all players.

        US Soccer Federation approach:
        - Each player connects to all teammates
        - Ball is added as an additional node
        - All players connect to the ball node

        This creates a star topology with the ball at the center, plus team connections.

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix (N_players+1, N_players+1) with ball as last node
        """
        n = len(features_df)
        # Add one extra node for the ball
        adjacency = np.zeros((n + 1, n + 1), dtype=np.float32)

        # Connect teammates (same as team strategy)
        teams = features_df['team'].values
        for i in range(n):
            for j in range(i + 1, n):
                if teams[i] == teams[j]:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0  # Undirected graph

        # Connect all players to ball node (index n)
        ball_idx = n
        for i in range(n):
            adjacency[i, ball_idx] = 1.0
            adjacency[ball_idx, i] = 1.0

        return csr_matrix(adjacency)

    def _build_distance_based_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build distance-based adjacency: Connect players within threshold.

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix
        """
        n = len(features_df)
        adjacency = np.zeros((n, n), dtype=np.float32)

        # Extract positions
        positions = features_df[['x', 'y']].values

        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= self.distance_threshold:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        return csr_matrix(adjacency)

    def _build_delaunay_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build Delaunay triangulation-based adjacency.

        Uses spatial triangulation to create natural connectivity based on player positions.

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix
        """
        n = len(features_df)
        adjacency = np.zeros((n, n), dtype=np.float32)

        # Extract positions
        positions = features_df[['x', 'y']].values

        # Perform Delaunay triangulation
        try:
            tri = Delaunay(positions)

            # Extract edges from simplices (triangles)
            for simplex in tri.simplices:
                # Each simplex is a triangle with 3 vertices
                for i in range(3):
                    v1 = simplex[i]
                    v2 = simplex[(i + 1) % 3]
                    adjacency[v1, v2] = 1.0
                    adjacency[v2, v1] = 1.0
        except Exception as e:
            warnings.warn(f"Delaunay triangulation failed: {e}. Falling back to distance-based.")
            return self._build_distance_based_adjacency(features_df)

        return csr_matrix(adjacency)

    def _build_ball_centric_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build ball-centric adjacency: Connect players near ball trajectory.

        Focuses on players within radius of ball landing zone.

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix
        """
        n = len(features_df)
        adjacency = np.zeros((n, n), dtype=np.float32)

        # Identify players near ball (using distance_to_ball_target feature)
        near_ball_mask = features_df['distance_to_ball_target'].values <= self.ball_zone_radius
        near_ball_indices = np.where(near_ball_mask)[0]

        # Connect all players near ball
        for i in near_ball_indices:
            for j in near_ball_indices:
                if i != j:
                    adjacency[i, j] = 1.0

        # Also connect players near ball to their teammates (extended connectivity)
        teams = features_df['team'].values
        for i in near_ball_indices:
            for j in range(n):
                if teams[i] == teams[j] and i != j:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        return csr_matrix(adjacency)

    def _build_zone_based_adjacency(self, features_df: pd.DataFrame) -> csr_matrix:
        """
        Build zone-based adjacency: Connect players in same tactical zones.

        Uses predefined tactical zones (6-yard box, penalty area, etc.)

        Args:
            features_df: DataFrame with player features

        Returns:
            Sparse adjacency matrix
        """
        n = len(features_df)
        adjacency = np.zeros((n, n), dtype=np.float32)

        # Extract positions
        positions = features_df[['x', 'y']].values

        # Assign players to zones
        player_zones = []
        for i in range(n):
            x, y = positions[i]
            zones = []
            for zone_name, (x_min, x_max, y_min, y_max) in TACTICAL_ZONES.items():
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    zones.append(zone_name)
            player_zones.append(set(zones))

        # Connect players in overlapping zones
        for i in range(n):
            for j in range(i + 1, n):
                if player_zones[i] & player_zones[j]:  # Intersection of zone sets
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        # Ensure minimum connectivity: connect teammates if no zone overlap
        teams = features_df['team'].values
        for i in range(n):
            if adjacency[i].sum() == 0:  # Isolated node
                for j in range(n):
                    if teams[i] == teams[j] and i != j:
                        adjacency[i, j] = 1.0
                        adjacency[j, i] = 1.0

        return csr_matrix(adjacency)

    def _compute_edge_features(
        self,
        features_df: pd.DataFrame,
        adjacency_matrix: csr_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute edge features for all connected pairs.

        Args:
            features_df: DataFrame with player features
            adjacency_matrix: Sparse adjacency matrix

        Returns:
            Tuple of (edge_index, edge_features)
            - edge_index: (2, N_edges) array of (source, target) node indices
            - edge_features: (N_edges, 6) array of edge feature vectors
        """
        # Convert sparse matrix to COO format for edge iteration
        adjacency_coo = adjacency_matrix.tocoo()

        edge_list = []
        edge_feature_list = []

        # Extract node data
        positions = features_df[['x', 'y']].values
        velocities = features_df[['vx', 'vy']].values

        # Check if we have a ball node (team_with_ball strategy)
        n_players = len(features_df)
        has_ball_node = (adjacency_matrix.shape[0] == n_players + 1)

        if has_ball_node:
            # Add ball position (use corner kick location or ball target)
            # Use first player's ball target as approximation
            ball_x = features_df.iloc[0]['x']  # Corner kick x location
            ball_y = features_df.iloc[0]['y']  # Corner kick y location
            ball_pos = np.array([[ball_x, ball_y]])
            ball_vel = np.array([[0.0, 0.0]])  # Ball has no velocity in freeze frame

            positions = np.vstack([positions, ball_pos])
            velocities = np.vstack([velocities, ball_vel])

        # Iterate over edges
        for i, j in zip(adjacency_coo.row, adjacency_coo.col):
            if i < j:  # Only process each edge once (undirected)
                # Calculate edge features
                pos_i, pos_j = positions[i], positions[j]
                vel_i, vel_j = velocities[i], velocities[j]

                # Distance (normalized by pitch diagonal)
                distance = np.linalg.norm(pos_i - pos_j)
                normalized_distance = distance / np.sqrt(PITCH_LENGTH**2 + PITCH_WIDTH**2)

                # Relative velocity
                relative_vx = vel_j[0] - vel_i[0]
                relative_vy = vel_j[1] - vel_i[1]

                # Angle between players (sine and cosine encoding)
                dx = pos_j[0] - pos_i[0]
                dy = pos_j[1] - pos_i[1]
                angle = np.arctan2(dy, dx)
                angle_sin = np.sin(angle)
                angle_cos = np.cos(angle)

                edge_features = EdgeFeatures(
                    distance=normalized_distance,
                    relative_vx=relative_vx,
                    relative_vy=relative_vy,
                    angle_sin=angle_sin,
                    angle_cos=angle_cos
                )

                # Add both directions (undirected graph represented as bidirectional)
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_feature_list.append(edge_features.to_array())
                edge_feature_list.append(edge_features.to_array())

        edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0), dtype=np.int32)
        edge_features = np.array(edge_feature_list) if edge_feature_list else np.zeros((0, 6), dtype=np.float32)

        return edge_index, edge_features


def compare_adjacency_strategies(features_df: pd.DataFrame, corner_id: str) -> Dict[str, CornerGraph]:
    """
    Build graphs with all adjacency strategies for comparison.

    Args:
        features_df: DataFrame with player features
        corner_id: Corner identifier

    Returns:
        Dictionary mapping strategy name to CornerGraph
    """
    strategies: List[AdjacencyStrategy] = ['team', 'team_with_ball', 'distance', 'delaunay', 'ball_centric', 'zone']
    graphs = {}

    for strategy in strategies:
        builder = GraphBuilder(adjacency_strategy=strategy)
        graphs[strategy] = builder.build_graph_from_features(features_df, corner_id)

    return graphs


if __name__ == "__main__":
    # Example usage
    print("Graph Builder Module - Phase 2.2")
    print("Adjacency strategies: team, team_with_ball, distance, delaunay, ball_centric, zone")
