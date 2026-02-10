"""Graph construction utilities for converting tracking frames to PyTorch Geometric graphs.

This module converts football/soccer tracking data into graph representations
suitable for Graph Neural Networks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from scipy.spatial.distance import cdist
from torch_geometric.data import Data


def frame_to_graph(
    frame,
    velocities: Dict[str, Dict[str, float]],
    corner_event,
    k_neighbors: int = 4,
) -> Data:
    """
    Convert a single tracking frame into a PyTorch Geometric graph.

    Args:
        frame: Tracking frame with player positions (kloppy Frame object)
        velocities: Dict mapping player_id -> {"vx": float, "vy": float}
        corner_event: The corner kick event (to identify attacking team)
        k_neighbors: Number of nearest neighbors for kNN edges

    Returns:
        torch_geometric.data.Data object with:
            - x: Node features tensor (num_nodes, 8)
            - edge_index: Edge connectivity tensor (2, num_edges)
            - pos: Node positions tensor (num_nodes, 2)
    """
    node_features = []
    positions = []
    player_ids = []
    teams = []

    # Goal position (standard pitch: 105m x 68m, goal at x=105, y=34)
    goal_x, goal_y = 105.0, 34.0

    # Ball position
    if frame.ball_coordinates is not None:
        ball_x = frame.ball_coordinates.x
        ball_y = frame.ball_coordinates.y
    else:
        ball_x, ball_y = 52.5, 34.0  # Center of pitch as fallback

    # Identify attacking team from corner event
    attacking_team = corner_event.team if hasattr(corner_event, "team") else None

    for player_id, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue

        x, y = pdata.coordinates.x, pdata.coordinates.y
        vel = velocities.get(player_id, {"vx": 0.0, "vy": 0.0})
        vx = vel.get("vx", 0.0)
        vy = vel.get("vy", 0.0)

        # Determine team flag
        if attacking_team is not None and hasattr(pdata, "team"):
            is_attacking = 1.0 if pdata.team == attacking_team else 0.0
        else:
            # Fallback: use team attribute if available
            is_attacking = 0.5  # Unknown team

        is_kicker_flag = 0.0  # Could be set based on event player if available
        dist_to_goal = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
        dist_to_ball = np.sqrt((x - ball_x) ** 2 + (y - ball_y) ** 2)

        node_features.append(
            [x, y, vx, vy, is_attacking, is_kicker_flag, dist_to_goal, dist_to_ball]
        )
        positions.append([x, y])
        player_ids.append(player_id)
        teams.append(is_attacking)

    # Add ball as node (with special team_flag = -1.0)
    dist_ball_to_goal = np.sqrt((ball_x - goal_x) ** 2 + (ball_y - goal_y) ** 2)
    node_features.append(
        [ball_x, ball_y, 0.0, 0.0, -1.0, 0.0, dist_ball_to_goal, 0.0]
    )
    positions.append([ball_x, ball_y])

    x_tensor = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    # --- Edge construction ---
    n_players = len(positions) - 1  # exclude ball for player edges
    edge_list = []

    if n_players > 0:
        player_pos = np.array(positions[:n_players])

        # 1. kNN proximity edges
        if n_players > k_neighbors:
            dist_matrix = cdist(player_pos, player_pos)
            np.fill_diagonal(dist_matrix, np.inf)

            for i in range(n_players):
                neighbors = np.argsort(dist_matrix[i])[:k_neighbors]
                for j in neighbors:
                    edge_list.append([i, j])

        # 2. Marking edges (nearest opponent)
        attackers = [i for i in range(n_players) if teams[i] == 1.0]
        defenders = [i for i in range(n_players) if teams[i] == 0.0]

        if attackers and defenders:
            att_pos = player_pos[attackers]
            def_pos = player_pos[defenders]
            cross_dist = cdist(att_pos, def_pos)

            # Each attacker -> nearest defender (bidirectional)
            for i, att_idx in enumerate(attackers):
                nearest_def = defenders[np.argmin(cross_dist[i])]
                edge_list.append([att_idx, nearest_def])
                edge_list.append([nearest_def, att_idx])

        # 3. Ball edges: connect ball to all players
        ball_idx = len(positions) - 1
        for i in range(n_players):
            edge_list.append([i, ball_idx])
            edge_list.append([ball_idx, i])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x_tensor, edge_index=edge_index, pos=pos)
