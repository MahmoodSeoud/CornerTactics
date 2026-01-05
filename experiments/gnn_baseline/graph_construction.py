"""Graph construction from StatsBomb freeze-frame data.

Converts corner kick freeze-frame data into PyTorch Geometric graph objects
for GNN-based outcome prediction.
"""

import math
from typing import List, Dict, Any, Literal

import torch
import numpy as np
from torch_geometric.data import Data


# Pitch dimensions for normalization (StatsBomb uses 120x80)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Goal position (center of goal line)
GOAL_X = 120.0
GOAL_Y = 40.0


def corner_to_graph(
    corner: Dict[str, Any],
    edge_type: Literal['knn', 'full'] = 'knn',
    k: int = 5,
) -> Data:
    """Convert a corner freeze-frame to a PyTorch Geometric graph.

    Args:
        corner: Corner data with 'freeze_frame', 'event', and 'shot_outcome'
        edge_type: Edge construction method ('knn' or 'full')
        k: Number of neighbors for k-NN (ignored if edge_type='full')

    Returns:
        PyTorch Geometric Data object with:
            - x: Node features [num_players, 5]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, 2]
            - y: Label (0 or 1)
    """
    freeze_frame = corner['freeze_frame']
    event = corner['event']
    ball_location = event['location']

    # Extract player data
    positions = []
    team_indicators = []
    keeper_indicators = []

    for player in freeze_frame:
        loc = player['location']
        positions.append([loc[0], loc[1]])
        # teammate=True means attacker (team taking the corner)
        team_indicators.append(1.0 if player['teammate'] else 0.0)
        keeper_indicators.append(1.0 if player['keeper'] else 0.0)

    positions = torch.tensor(positions, dtype=torch.float32)
    team_indicators = torch.tensor(team_indicators, dtype=torch.float32)
    keeper_indicators = torch.tensor(keeper_indicators, dtype=torch.float32)

    # Compute node features
    node_features = compute_node_features(
        positions, team_indicators, keeper_indicators, ball_location
    )

    # Create edges
    if edge_type == 'full':
        edge_index = create_full_edges(len(freeze_frame))
    else:
        edge_index = create_knn_edges(positions, k=k)

    # Compute edge features
    edge_attr = compute_edge_features(positions, edge_index)

    # Label
    y = torch.tensor([corner['shot_outcome']], dtype=torch.float32)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


def compute_node_features(
    positions: torch.Tensor,
    team_indicators: torch.Tensor,
    keeper_indicators: torch.Tensor,
    ball_location: List[float],
) -> torch.Tensor:
    """Compute node features for each player.

    Features:
        0: x position (normalized 0-1)
        1: y position (normalized 0-1)
        2: team indicator (1=attacker, 0=defender)
        3: distance to goal (normalized)
        4: distance to ball (normalized)

    Args:
        positions: [num_players, 2] raw positions
        team_indicators: [num_players] team flags
        keeper_indicators: [num_players] keeper flags
        ball_location: [x, y] ball position

    Returns:
        [num_players, 5] feature matrix
    """
    num_players = positions.shape[0]

    # Normalize positions to [0, 1]
    norm_x = positions[:, 0] / PITCH_LENGTH
    norm_y = positions[:, 1] / PITCH_WIDTH

    # Distance to goal (normalize by max possible distance)
    goal_pos = torch.tensor([GOAL_X, GOAL_Y], dtype=torch.float32)
    dist_to_goal = torch.norm(positions - goal_pos, dim=1)
    max_dist_goal = math.sqrt(PITCH_LENGTH**2 + PITCH_WIDTH**2)
    norm_dist_goal = dist_to_goal / max_dist_goal

    # Distance to ball
    ball_pos = torch.tensor(ball_location, dtype=torch.float32)
    dist_to_ball = torch.norm(positions - ball_pos, dim=1)
    norm_dist_ball = dist_to_ball / max_dist_goal

    # Stack features
    features = torch.stack([
        norm_x,
        norm_y,
        team_indicators,
        norm_dist_goal,
        norm_dist_ball,
    ], dim=1)

    return features


def create_knn_edges(
    positions: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Create k-nearest neighbor edges.

    Args:
        positions: [num_nodes, 2] node positions
        k: Number of neighbors per node

    Returns:
        [2, num_edges] edge index (directed edges)
    """
    num_nodes = positions.shape[0]
    k = min(k, num_nodes - 1)  # Can't have more neighbors than other nodes

    # Compute pairwise distances
    # diff[i, j] = positions[i] - positions[j]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist_matrix = torch.norm(diff, dim=2)

    # Set diagonal to infinity to exclude self-loops
    dist_matrix.fill_diagonal_(float('inf'))

    # Find k nearest neighbors for each node
    _, indices = torch.topk(dist_matrix, k, largest=False, dim=1)

    # Build edge index
    sources = torch.arange(num_nodes).unsqueeze(1).expand(-1, k).flatten()
    targets = indices.flatten()

    edge_index = torch.stack([sources, targets], dim=0)

    return edge_index


def create_full_edges(num_nodes: int) -> torch.Tensor:
    """Create fully connected edges (no self-loops).

    Args:
        num_nodes: Number of nodes

    Returns:
        [2, num_edges] edge index
    """
    sources = []
    targets = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)

    return torch.tensor([sources, targets], dtype=torch.long)


def compute_edge_features(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute edge features.

    Features:
        0: Euclidean distance (normalized)
        1: Angle to goal (normalized to [-1, 1])

    Args:
        positions: [num_nodes, 2] node positions
        edge_index: [2, num_edges] edge connectivity

    Returns:
        [num_edges, 2] edge features
    """
    sources, targets = edge_index[0], edge_index[1]

    # Get source and target positions
    src_pos = positions[sources]
    tgt_pos = positions[targets]

    # Distance between connected nodes
    dist = torch.norm(tgt_pos - src_pos, dim=1)
    max_dist = math.sqrt(PITCH_LENGTH**2 + PITCH_WIDTH**2)
    norm_dist = dist / max_dist

    # Angle from source to goal
    goal_pos = torch.tensor([GOAL_X, GOAL_Y], dtype=torch.float32)
    vec_to_goal = goal_pos - src_pos
    angles = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    norm_angles = angles / math.pi  # Normalize to [-1, 1]

    edge_attr = torch.stack([norm_dist, norm_angles], dim=1)

    return edge_attr


def build_graph_dataset(
    corners: List[Dict[str, Any]],
    edge_type: Literal['knn', 'full'] = 'knn',
    k: int = 5,
) -> List[Data]:
    """Build a dataset of graphs from corner data.

    Args:
        corners: List of corner dictionaries
        edge_type: Edge construction method
        k: Number of neighbors for k-NN

    Returns:
        List of PyTorch Geometric Data objects
    """
    dataset = []
    for corner in corners:
        graph = corner_to_graph(corner, edge_type=edge_type, k=k)
        dataset.append(graph)
    return dataset
