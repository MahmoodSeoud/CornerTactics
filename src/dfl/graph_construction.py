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


def corner_to_temporal_graphs(
    tracking_dataset,
    corner_event,
    fps: int = 25,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
) -> List[Data]:
    """
    Convert a corner kick into a sequence of graphs over time.

    Args:
        tracking_dataset: kloppy TrackingDataset
        corner_event: The corner kick event
        fps: Frame rate of tracking data
        pre_seconds: Seconds before corner to include
        post_seconds: Seconds after corner to include

    Returns:
        List of Data objects, one per frame, with added attributes:
            - frame_idx: Index within the sequence
            - relative_time: Time relative to corner delivery (seconds)
    """
    from src.dfl.data_loading import extract_corner_sequence, compute_velocities

    # Extract frames around the corner
    frames = extract_corner_sequence(
        tracking_dataset,
        corner_event,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )

    if not frames:
        return []

    # Compute velocities for all frames
    velocities = compute_velocities(frames, fps=fps)

    # Convert each frame to a graph
    graphs = []
    for i, frame in enumerate(frames):
        graph = frame_to_graph(
            frame=frame,
            velocities=velocities[i],
            corner_event=corner_event,
        )
        graph.frame_idx = i
        graph.relative_time = (i / fps) - pre_seconds
        graphs.append(graph)

    return graphs


def label_corner(
    corner_event,
    event_dataset,
    n_subsequent_events: int = 5,
) -> Dict[str, Any]:
    """
    Create multi-head labels for a corner kick based on subsequent events.

    Args:
        corner_event: The corner kick event
        event_dataset: kloppy EventDataset containing all events
        n_subsequent_events: Number of events after corner to analyze

    Returns:
        Dict with labels:
            - shot_binary: 1 if shot within next n events, else 0
            - goal_binary: 1 if goal within next n events, else 0
            - first_contact_team: 'attacking', 'defending', or 'unknown'
            - outcome_class: One of ['goal', 'shot_saved', 'shot_blocked',
                                      'clearance', 'ball_receipt', 'other']
    """
    all_events = list(event_dataset.events)

    # Find the corner event in the event list
    corner_idx = None
    corner_timestamp = corner_event.timestamp

    for i, e in enumerate(all_events):
        if e.timestamp == corner_timestamp:
            # Check if it's the same event (same type and timestamp)
            if str(e.event_type).lower() == str(corner_event.event_type).lower():
                corner_idx = i
                break

    # If exact match not found, find by closest timestamp
    if corner_idx is None:
        min_diff = float("inf")
        for i, e in enumerate(all_events):
            diff = abs(_get_timestamp_seconds(e.timestamp) - _get_timestamp_seconds(corner_timestamp))
            if diff < min_diff:
                min_diff = diff
                corner_idx = i

    labels = {
        "shot_binary": 0,
        "goal_binary": 0,
        "first_contact_team": "unknown",
        "outcome_class": "other",
    }

    if corner_idx is None:
        return labels

    # Get events following this corner
    subsequent = all_events[corner_idx + 1 : corner_idx + 1 + n_subsequent_events]

    for event in subsequent:
        event_type_str = str(event.event_type).lower()

        # Check for shots
        if "shot" in event_type_str:
            labels["shot_binary"] = 1
            labels["outcome_class"] = "shot_saved"  # Default shot outcome

            # Check if goal
            if "goal" in event_type_str:
                labels["goal_binary"] = 1
                labels["outcome_class"] = "goal"
            elif hasattr(event, "result"):
                result_str = str(event.result).lower() if event.result else ""
                if "goal" in result_str:
                    labels["goal_binary"] = 1
                    labels["outcome_class"] = "goal"
                elif "blocked" in result_str:
                    labels["outcome_class"] = "shot_blocked"

        # Check for clearance
        if "clearance" in event_type_str:
            if labels["outcome_class"] == "other":
                labels["outcome_class"] = "clearance"

        # First contact detection
        if labels["first_contact_team"] == "unknown":
            if hasattr(event, "team") and event.team is not None:
                if hasattr(corner_event, "team") and corner_event.team is not None:
                    if event.team == corner_event.team:
                        labels["first_contact_team"] = "attacking"
                    else:
                        labels["first_contact_team"] = "defending"

    return labels


def _get_timestamp_seconds(timestamp) -> float:
    """Convert timestamp to seconds."""
    if hasattr(timestamp, "total_seconds"):
        return timestamp.total_seconds()
    return float(timestamp)


def build_corner_dataset_from_match(
    tracking_dataset,
    event_dataset,
    match_id: str,
    fps: int = 25,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Build corner kick dataset from a single match.

    Args:
        tracking_dataset: kloppy TrackingDataset
        event_dataset: kloppy EventDataset
        match_id: Identifier for this match
        fps: Frame rate of tracking data
        pre_seconds: Seconds before corner to include
        post_seconds: Seconds after corner to include

    Returns:
        List of dicts, each containing:
            - 'graphs': List[Data] (temporal sequence of graphs)
            - 'labels': dict (multi-head labels)
            - 'match_id': str
            - 'corner_time': float (timestamp of corner)
    """
    from src.dfl.data_loading import find_corner_events

    corners = find_corner_events(event_dataset)
    dataset = []

    for corner in corners:
        graphs = corner_to_temporal_graphs(
            tracking_dataset,
            corner,
            fps=fps,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )

        labels = label_corner(corner, event_dataset)

        if graphs and labels:
            corner_time = _get_timestamp_seconds(corner.timestamp)
            dataset.append(
                {
                    "graphs": graphs,
                    "labels": labels,
                    "match_id": match_id,
                    "corner_time": corner_time,
                }
            )

    return dataset
