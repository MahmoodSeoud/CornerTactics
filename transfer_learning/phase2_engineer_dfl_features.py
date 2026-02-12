#!/usr/bin/env python3
"""
Phase 2: Engineer DFL Corner Features to Match USSF Schema
==========================================================

Transforms DFL corner kick graphs to be compatible with the USSF-pretrained
backbone for transfer learning.

USSF Node Features (12):
    0. x                   - [0, 1] normalized pitch position
    1. y                   - [0, 1] normalized pitch position
    2. vx                  - Unit vector x component (direction)
    3. vy                  - Unit vector y component (direction)
    4. velocity_mag        - [0, 1] normalized speed
    5. velocity_angle      - [0, 1] normalized angle from atan2
    6. dist_goal           - [0, 1] normalized distance to goal
    7. angle_goal          - [0, 1] normalized angle to goal
    8. dist_ball           - [0, 1] normalized distance to ball
    9. angle_ball          - [0, 1] normalized angle to ball
    10. attacking_team_flag - Binary 0/1
    11. potential_receiver  - Binary 0/1 (set to 0 for corner kicks)

USSF Edge Features (6):
    0. player_distance       - Normalized Euclidean distance
    1. speed_difference      - Signed speed difference [-1, 1]
    2. positional_sine_angle - (sin(angle) + 1) / 2, range [0, 1]
    3. positional_cosine_angle - (cos(angle) + 1) / 2, range [0, 1]
    4. velocity_sine_angle   - (sin(angle) + 1) / 2, range [0, 1]
    5. velocity_cosine_angle - (cos(angle) + 1) / 2, range [0, 1]
"""

import argparse
import pickle
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch_geometric.data import Data
from scipy import stats
from scipy.spatial.distance import cdist

# Configuration
DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
CORNER_DATASET = Path("/home/mseo/CornerTactics/results/ablation/corner_dataset.pkl")
USSF_DATA = DATA_DIR / "combined.pkl"

# DFL pitch dimensions (meters)
DFL_PITCH_LENGTH = 105.0
DFL_PITCH_WIDTH = 68.0

# USSF normalization constants (estimated from data analysis)
USSF_MAX_VELOCITY = 10.0  # m/s - typical max sprint speed
USSF_MAX_DISTANCE = 1.0  # Already in normalized coordinates


def normalize_coordinates(x: float, y: float) -> Tuple[float, float]:
    """Normalize DFL coordinates to [0, 1] range."""
    x_norm = np.clip(x / DFL_PITCH_LENGTH, 0.0, 1.0)
    y_norm = np.clip(y / DFL_PITCH_WIDTH, 0.0, 1.0)
    return x_norm, y_norm


def compute_unit_vector(vx: float, vy: float) -> Tuple[float, float, float]:
    """
    Convert velocity to unit vector and magnitude.

    Returns:
        (vx_unit, vy_unit, magnitude)
    """
    magnitude = np.sqrt(vx**2 + vy**2)
    if magnitude < 1e-8:
        # Zero velocity - return arbitrary direction
        return 1.0, 0.0, 0.0
    return vx / magnitude, vy / magnitude, magnitude


def normalize_angle(angle_rad: float) -> float:
    """Normalize angle from [-pi, pi] to [0, 1]."""
    return (angle_rad + np.pi) / (2 * np.pi)


def transform_dfl_node_to_ussf(
    x_norm: float, y_norm: float,
    vx: float, vy: float,
    is_attacking: float,
    is_ball: bool,
    goal_x_norm: float, goal_y_norm: float,
    ball_x_norm: float, ball_y_norm: float,
    flip_x: bool = True,
    max_velocity: float = USSF_MAX_VELOCITY
) -> np.ndarray:
    """
    Transform a single DFL node to USSF feature format.

    Note: DFL coordinates are already normalized [0,1] but with goal at x=0.
    We flip x to match USSF convention (goal at x=1, attacking direction positive x).

    Args:
        x_norm, y_norm: Normalized DFL coordinates [0,1]
        vx, vy: DFL velocities (m/s after scaling)
        is_attacking: 1.0 for attacking, 0.0 for defending, 0.5 for unknown, -1.0 for ball
        is_ball: Whether this node is the ball
        goal_x_norm, goal_y_norm: Normalized goal position (after flip if applicable)
        ball_x_norm, ball_y_norm: Normalized ball position (after flip if applicable)
        flip_x: Whether to flip x coordinate to match USSF convention
        max_velocity: Maximum velocity for normalization

    Returns:
        12-dimensional feature vector matching USSF schema
    """
    # Flip x coordinate to match USSF convention (goal at x=1)
    if flip_x:
        x_final = 1.0 - x_norm
        vx = -vx  # Flip velocity direction too
    else:
        x_final = x_norm
    y_final = y_norm

    # Clip to valid range
    x_final = np.clip(x_final, 0.0, 1.0)
    y_final = np.clip(y_final, 0.0, 1.0)

    # Velocity: convert to unit vector + normalized magnitude
    vx_unit, vy_unit, vel_mag = compute_unit_vector(vx, vy)
    vel_mag_norm = np.clip(vel_mag / max_velocity, 0.0, 1.0)

    # Velocity angle (direction of movement)
    vel_angle = np.arctan2(vy_unit, vx_unit)
    vel_angle_norm = normalize_angle(vel_angle)

    # Distance and angle to goal
    dx_goal = goal_x_norm - x_final
    dy_goal = goal_y_norm - y_final
    dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
    dist_goal_norm = np.clip(dist_goal / np.sqrt(2), 0.0, 1.0)  # Normalize by pitch diagonal
    angle_goal = np.arctan2(dy_goal, dx_goal)
    angle_goal_norm = normalize_angle(angle_goal)

    # Distance and angle to ball
    dx_ball = ball_x_norm - x_final
    dy_ball = ball_y_norm - y_final
    dist_ball = np.sqrt(dx_ball**2 + dy_ball**2)
    dist_ball_norm = np.clip(dist_ball / np.sqrt(2), 0.0, 1.0)
    angle_ball = np.arctan2(dy_ball, dx_ball)
    angle_ball_norm = normalize_angle(angle_ball)

    # Attacking team flag
    # DFL data has is_attacking = 0.5 (unknown) for all players, -1.0 for ball
    # Use heuristic: players near the goal are more likely defending
    # For now, set based on position - players with x > 0.7 (after flip) are likely attacking
    if is_ball:
        attacking_flag = 0.0  # Ball is neutral
    elif is_attacking < -0.5:  # Ball marker
        attacking_flag = 0.0
    elif is_attacking > 0.6:  # Actually known to be attacking
        attacking_flag = 1.0
    elif is_attacking < 0.4 and is_attacking > -0.5:  # Actually known to be defending
        attacking_flag = 0.0
    else:  # Unknown (0.5) - use position heuristic
        # In corner kicks, attacking players are typically further from goal
        # After x flip, goal is at x=1, so lower x values = further from goal = likely attacking
        attacking_flag = 1.0 if x_final < 0.5 else 0.0

    # Potential receiver: set to 0 for all corner kick players
    # (This feature is USSF-specific for counterattacks)
    potential_receiver = 0.0

    return np.array([
        x_final,            # 0. x
        y_final,            # 1. y
        vx_unit,            # 2. vx (unit vector)
        vy_unit,            # 3. vy (unit vector)
        vel_mag_norm,       # 4. velocity_mag
        vel_angle_norm,     # 5. velocity_angle
        dist_goal_norm,     # 6. dist_goal
        angle_goal_norm,    # 7. angle_goal
        dist_ball_norm,     # 8. dist_ball
        angle_ball_norm,    # 9. angle_ball
        attacking_flag,     # 10. attacking_team_flag
        potential_receiver, # 11. potential_receiver
    ], dtype=np.float32)


def compute_edge_features(
    node_i: np.ndarray,
    node_j: np.ndarray
) -> np.ndarray:
    """
    Compute edge features between two nodes.

    Args:
        node_i, node_j: 12-dimensional node feature vectors

    Returns:
        6-dimensional edge feature vector
    """
    # Extract relevant features
    xi, yi = node_i[0], node_i[1]
    xj, yj = node_j[0], node_j[1]
    vxi, vyi = node_i[2], node_i[3]  # Unit vectors
    vxj, vyj = node_j[2], node_j[3]
    vel_mag_i = node_i[4]
    vel_mag_j = node_j[4]

    # 1. Player distance (normalized by pitch diagonal)
    dx = xj - xi
    dy = yj - yi
    player_distance = np.sqrt(dx**2 + dy**2)
    player_distance_norm = np.clip(player_distance / np.sqrt(2), 0.0, 1.0)

    # 2. Speed difference (signed, normalized to [-1, 1])
    speed_diff = vel_mag_j - vel_mag_i  # Already in [0,1] range

    # 3-4. Positional angle features
    pos_angle = np.arctan2(dy, dx)
    positional_sine = (np.sin(pos_angle) + 1) / 2  # [0, 1]
    positional_cosine = (np.cos(pos_angle) + 1) / 2  # [0, 1]

    # 5-6. Velocity angle features (angle between velocity vectors)
    # Compute angle between the two unit velocity vectors
    dot_product = vxi * vxj + vyi * vyj
    cross_product = vxi * vyj - vyi * vxj
    vel_angle = np.arctan2(cross_product, dot_product)
    velocity_sine = (np.sin(vel_angle) + 1) / 2  # [0, 1]
    velocity_cosine = (np.cos(vel_angle) + 1) / 2  # [0, 1]

    return np.array([
        player_distance_norm,  # 0. player_distance
        speed_diff,            # 1. speed_difference
        positional_sine,       # 2. positional_sine_angle
        positional_cosine,     # 3. positional_cosine_angle
        velocity_sine,         # 4. velocity_sine_angle
        velocity_cosine,       # 5. velocity_cosine_angle
    ], dtype=np.float32)


def build_dense_adjacency(n_nodes: int) -> torch.Tensor:
    """Build fully connected (dense) edge index."""
    # Create all pairs (i, j) where i != j
    src = []
    dst = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def build_normal_adjacency(
    nodes: np.ndarray,
    k_neighbors: int = 4
) -> torch.Tensor:
    """
    Build normal adjacency: kNN + team-based connectivity through ball.

    Similar to USSF's 'normal' adjacency type.
    """
    n_nodes = len(nodes)
    edge_set = set()

    # Separate players and ball
    player_indices = []
    ball_idx = None

    for i in range(n_nodes):
        # Ball has attacking_flag = 0 and all distances = 0 typically
        # Or we can check if dist_ball (feature 8) is very small
        if nodes[i, 8] < 0.01:  # This is the ball (dist_ball ≈ 0)
            ball_idx = i
        else:
            player_indices.append(i)

    if not player_indices:
        return torch.empty((2, 0), dtype=torch.long)

    n_players = len(player_indices)

    # 1. kNN edges among players
    if n_players > k_neighbors:
        player_pos = nodes[player_indices, :2]  # x, y positions
        dist_matrix = cdist(player_pos, player_pos)
        np.fill_diagonal(dist_matrix, np.inf)

        for i, pi in enumerate(player_indices):
            neighbors = np.argsort(dist_matrix[i])[:k_neighbors]
            for j in neighbors:
                pj = player_indices[j]
                edge_set.add((pi, pj))
    else:
        # Fully connect players if fewer than k
        for pi in player_indices:
            for pj in player_indices:
                if pi != pj:
                    edge_set.add((pi, pj))

    # 2. Ball edges: connect ball to all players
    if ball_idx is not None:
        for pi in player_indices:
            edge_set.add((pi, ball_idx))
            edge_set.add((ball_idx, pi))

    # 3. Team connectivity (attackers to defenders)
    attackers = [i for i in player_indices if nodes[i, 10] > 0.5]
    defenders = [i for i in player_indices if nodes[i, 10] < 0.5]

    if attackers and defenders:
        att_pos = nodes[attackers, :2]
        def_pos = nodes[defenders, :2]
        cross_dist = cdist(att_pos, def_pos)

        # Each attacker connects to nearest defender
        for i, att_idx in enumerate(attackers):
            nearest_def_local = np.argmin(cross_dist[i])
            nearest_def = defenders[nearest_def_local]
            edge_set.add((att_idx, nearest_def))
            edge_set.add((nearest_def, att_idx))

    # Convert to tensor
    if edge_set:
        edges = list(edge_set)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        return torch.tensor([src, dst], dtype=torch.long)
    else:
        return torch.empty((2, 0), dtype=torch.long)


def transform_dfl_graph(
    graph: Data,
    adj_type: str = 'dense'
) -> Data:
    """
    Transform a single DFL graph to USSF-compatible format.

    DFL coordinates are already normalized [0,1] with goal at x=0.
    We flip x to match USSF convention (goal at x=1).

    Args:
        graph: PyG Data object with DFL features
        adj_type: 'dense' or 'normal' adjacency type

    Returns:
        PyG Data object with USSF-compatible features
    """
    # Get original features
    x_orig = graph.x.numpy() if hasattr(graph.x, 'numpy') else np.array(graph.x)
    n_nodes = x_orig.shape[0]

    # DFL features: [x, y, vx, vy, is_attacking, is_kicker, dist_goal, dist_ball]
    # Positions are already normalized [0,1], velocities are in m/frame

    # Find ball (is_attacking = -1.0)
    ball_idx = None
    for i in range(n_nodes):
        if x_orig[i, 4] < -0.5:  # is_attacking = -1.0 for ball
            ball_idx = i
            break

    # Get ball position (already normalized, but we need to flip x)
    if ball_idx is not None:
        ball_x_dfl = x_orig[ball_idx, 0]  # Already normalized
        ball_y_dfl = x_orig[ball_idx, 1]
    else:
        # Fallback - corner kicks typically have ball near x=0
        ball_x_dfl = 0.05
        ball_y_dfl = 0.5

    # After flip: ball_x_norm = 1 - ball_x_dfl (ball will be near x=0.95)
    ball_x_norm = 1.0 - ball_x_dfl
    ball_y_norm = ball_y_dfl

    # Goal position after flip - goal is at x=1, y=0.5
    goal_x_norm = 1.0
    goal_y_norm = 0.5

    # Transform each node
    new_features = []
    for i in range(n_nodes):
        x_dfl = x_orig[i, 0]  # Already normalized
        y_dfl = x_orig[i, 1]  # Already normalized
        vx = x_orig[i, 2]     # m/frame
        vy = x_orig[i, 3]     # m/frame
        is_attacking = x_orig[i, 4]
        is_ball = (i == ball_idx)

        # Scale velocities (DFL velocities are in m/frame, convert to m/s)
        # At 25 fps, velocity in m/s = velocity_per_frame * 25
        vx_mps = vx * 25.0
        vy_mps = vy * 25.0

        node_features = transform_dfl_node_to_ussf(
            x_norm=x_dfl,
            y_norm=y_dfl,
            vx=vx_mps,
            vy=vy_mps,
            is_attacking=is_attacking,
            is_ball=is_ball,
            goal_x_norm=goal_x_norm,
            goal_y_norm=goal_y_norm,
            ball_x_norm=ball_x_norm,
            ball_y_norm=ball_y_norm,
            flip_x=True
        )
        new_features.append(node_features)

    new_x = np.array(new_features, dtype=np.float32)

    # Build adjacency
    if adj_type == 'dense':
        edge_index = build_dense_adjacency(n_nodes)
    else:
        edge_index = build_normal_adjacency(new_x)

    # Compute edge features
    n_edges = edge_index.shape[1]
    edge_attr = []
    for e in range(n_edges):
        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
        edge_feat = compute_edge_features(new_x[src], new_x[dst])
        edge_attr.append(edge_feat)

    if edge_attr:
        edge_attr = np.array(edge_attr, dtype=np.float32)
    else:
        edge_attr = np.zeros((0, 6), dtype=np.float32)

    # Create new Data object
    new_graph = Data(
        x=torch.tensor(new_x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        pos=torch.tensor(new_x[:, :2], dtype=torch.float32),  # Normalized positions
    )

    # Copy over metadata
    if hasattr(graph, 'frame_idx'):
        new_graph.frame_idx = graph.frame_idx
    if hasattr(graph, 'relative_time'):
        new_graph.relative_time = graph.relative_time

    return new_graph


def transform_corner_dataset(
    dataset: List[Dict],
    adj_type: str = 'dense',
    frame_selection: str = 'delivery'
) -> List[Dict]:
    """
    Transform entire corner dataset to USSF-compatible format.

    Args:
        dataset: List of corner samples with 'graphs' and 'labels'
        adj_type: 'dense' or 'normal' adjacency type
        frame_selection: 'delivery' (t=0), 'all', or 'sample'

    Returns:
        Transformed dataset
    """
    transformed = []

    for i, sample in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"  Transforming corner {i+1}/{len(dataset)}...")

        graphs = sample['graphs']

        # Select frames
        if frame_selection == 'delivery':
            # Find frame closest to t=0 (delivery moment)
            delivery_idx = None
            for j, g in enumerate(graphs):
                if hasattr(g, 'relative_time') and abs(g.relative_time) < 0.1:
                    delivery_idx = j
                    break
            if delivery_idx is None:
                delivery_idx = len(graphs) // 3  # Approximately 2s into 8s window

            selected_graphs = [graphs[delivery_idx]]
        elif frame_selection == 'sample':
            # Sample frames: before, at, and after delivery
            indices = [0, len(graphs)//3, 2*len(graphs)//3, -1]
            selected_graphs = [graphs[i] for i in indices if i < len(graphs)]
        else:  # 'all'
            selected_graphs = graphs

        # Transform selected graphs
        transformed_graphs = []
        for g in selected_graphs:
            try:
                tg = transform_dfl_graph(g, adj_type=adj_type)
                transformed_graphs.append(tg)
            except Exception as e:
                print(f"    Warning: Failed to transform graph: {e}")
                continue

        if transformed_graphs:
            transformed.append({
                'graphs': transformed_graphs,
                'labels': sample['labels'],
                'match_id': sample['match_id'],
                'corner_time': sample['corner_time']
            })

    return transformed


def compute_feature_statistics(dataset: List[Dict]) -> Dict:
    """Compute feature statistics across all graphs."""
    all_nodes = []
    all_edges = []

    for sample in dataset:
        for graph in sample['graphs']:
            x = graph.x.numpy() if hasattr(graph.x, 'numpy') else np.array(graph.x)
            all_nodes.append(x)

            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                e = graph.edge_attr
                e = e.numpy() if hasattr(e, 'numpy') else np.array(e)
                if len(e) > 0:
                    all_edges.append(e)

    all_nodes = np.concatenate(all_nodes, axis=0)
    all_edges = np.concatenate(all_edges, axis=0) if all_edges else np.zeros((0, 6))

    node_names = ['x', 'y', 'vx', 'vy', 'velocity_mag', 'velocity_angle',
                  'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball',
                  'attacking_team_flag', 'potential_receiver']

    edge_names = ['player_distance', 'speed_difference',
                  'positional_sine_angle', 'positional_cosine_angle',
                  'velocity_sine_angle', 'velocity_cosine_angle']

    stats_dict = {
        'node_features': {},
        'edge_features': {},
        'total_nodes': len(all_nodes),
        'total_edges': len(all_edges)
    }

    for i, name in enumerate(node_names):
        vals = all_nodes[:, i]
        stats_dict['node_features'][name] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max())
        }

    for i, name in enumerate(edge_names):
        if i < all_edges.shape[1]:
            vals = all_edges[:, i]
            stats_dict['edge_features'][name] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max())
            }

    return stats_dict


def run_ks_tests(
    dfl_dataset: List[Dict],
    ussf_data: Dict,
    adj_type: str = 'dense'
) -> Dict:
    """
    Run Kolmogorov-Smirnov tests comparing DFL and USSF distributions.

    Returns dict with KS statistics and p-values for each feature.
    """
    # Gather DFL features
    dfl_nodes = []
    dfl_edges = []

    for sample in dfl_dataset:
        for graph in sample['graphs']:
            x = graph.x.numpy() if hasattr(graph.x, 'numpy') else np.array(graph.x)
            dfl_nodes.append(x)

            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                e = graph.edge_attr
                e = e.numpy() if hasattr(e, 'numpy') else np.array(e)
                if len(e) > 0:
                    dfl_edges.append(e)

    dfl_nodes = np.concatenate(dfl_nodes, axis=0)
    dfl_edges = np.concatenate(dfl_edges, axis=0) if dfl_edges else np.zeros((0, 6))

    # Gather USSF features
    ussf_nodes = np.concatenate(ussf_data[adj_type]['x'], axis=0)
    ussf_edges = np.concatenate(ussf_data[adj_type]['e'], axis=0)

    # Run KS tests
    node_names = ['x', 'y', 'vx', 'vy', 'velocity_mag', 'velocity_angle',
                  'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball',
                  'attacking_team_flag', 'potential_receiver']

    edge_names = ['player_distance', 'speed_difference',
                  'positional_sine_angle', 'positional_cosine_angle',
                  'velocity_sine_angle', 'velocity_cosine_angle']

    results = {
        'node_features': {},
        'edge_features': {},
        'summary': {
            'significant_node_mismatches': [],
            'significant_edge_mismatches': []
        }
    }

    # Node feature KS tests
    for i, name in enumerate(node_names):
        dfl_vals = dfl_nodes[:, i]
        ussf_vals = ussf_nodes[:, i]

        ks_stat, p_val = stats.ks_2samp(dfl_vals, ussf_vals)

        results['node_features'][name] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_val),
            'significant': p_val < 0.01,
            'dfl_mean': float(dfl_vals.mean()),
            'dfl_std': float(dfl_vals.std()),
            'ussf_mean': float(ussf_vals.mean()),
            'ussf_std': float(ussf_vals.std())
        }

        if p_val < 0.01:
            results['summary']['significant_node_mismatches'].append(name)

    # Edge feature KS tests
    if len(dfl_edges) > 0:
        for i, name in enumerate(edge_names):
            if i < dfl_edges.shape[1] and i < ussf_edges.shape[1]:
                dfl_vals = dfl_edges[:, i]
                ussf_vals = ussf_edges[:, i]

                ks_stat, p_val = stats.ks_2samp(dfl_vals, ussf_vals)

                results['edge_features'][name] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.01,
                    'dfl_mean': float(dfl_vals.mean()),
                    'dfl_std': float(dfl_vals.std()),
                    'ussf_mean': float(ussf_vals.mean()),
                    'ussf_std': float(ussf_vals.std())
                }

                if p_val < 0.01:
                    results['summary']['significant_edge_mismatches'].append(name)

    return results


def generate_report(
    dfl_stats: Dict,
    ks_results: Dict,
    adj_type: str
) -> str:
    """Generate markdown report."""
    report = []
    report.append("# DFL to USSF Feature Distribution Comparison Report")
    report.append(f"\n**Adjacency Type:** {adj_type}")
    report.append(f"**Generated:** {datetime.now().isoformat()}")

    report.append("\n## 1. DFL Transformed Feature Statistics")
    report.append(f"\n**Total nodes:** {dfl_stats['total_nodes']}")
    report.append(f"**Total edges:** {dfl_stats['total_edges']}")

    report.append("\n### Node Features")
    report.append("\n| Feature | Mean | Std | Min | Max |")
    report.append("|---------|------|-----|-----|-----|")
    for name, s in dfl_stats['node_features'].items():
        report.append(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

    if dfl_stats['edge_features']:
        report.append("\n### Edge Features")
        report.append("\n| Feature | Mean | Std | Min | Max |")
        report.append("|---------|------|-----|-----|-----|")
        for name, s in dfl_stats['edge_features'].items():
            report.append(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

    report.append("\n## 2. Kolmogorov-Smirnov Test Results")
    report.append("\nTests comparing DFL transformed features vs USSF original features.")
    report.append("**Significance threshold:** p < 0.01")

    report.append("\n### Node Features")
    report.append("\n| Feature | KS Stat | p-value | Significant | DFL Mean | USSF Mean |")
    report.append("|---------|---------|---------|-------------|----------|-----------|")
    for name, r in ks_results['node_features'].items():
        sig = "**YES**" if r['significant'] else "No"
        report.append(f"| {name} | {r['ks_statistic']:.4f} | {r['p_value']:.4e} | {sig} | {r['dfl_mean']:.4f} | {r['ussf_mean']:.4f} |")

    if ks_results['edge_features']:
        report.append("\n### Edge Features")
        report.append("\n| Feature | KS Stat | p-value | Significant | DFL Mean | USSF Mean |")
        report.append("|---------|---------|---------|-------------|----------|-----------|")
        for name, r in ks_results['edge_features'].items():
            sig = "**YES**" if r['significant'] else "No"
            report.append(f"| {name} | {r['ks_statistic']:.4f} | {r['p_value']:.4e} | {sig} | {r['dfl_mean']:.4f} | {r['ussf_mean']:.4f} |")

    report.append("\n## 3. Summary")
    n_node_mismatch = len(ks_results['summary']['significant_node_mismatches'])
    n_edge_mismatch = len(ks_results['summary']['significant_edge_mismatches'])

    report.append(f"\n**Node features with significant distribution mismatch:** {n_node_mismatch}/12")
    if ks_results['summary']['significant_node_mismatches']:
        report.append(f"- {', '.join(ks_results['summary']['significant_node_mismatches'])}")

    report.append(f"\n**Edge features with significant distribution mismatch:** {n_edge_mismatch}/6")
    if ks_results['summary']['significant_edge_mismatches']:
        report.append(f"- {', '.join(ks_results['summary']['significant_edge_mismatches'])}")

    report.append("\n## 4. Interpretation")
    report.append("""
Features with significant distribution mismatch (p < 0.01) may transfer poorly.
This is expected due to:

1. **Different game situations:** Counterattacks (USSF) vs corner kicks (DFL)
   - Counterattacks: players spread across half-pitch, high velocities
   - Corner kicks: dense clustering in penalty area, stationary setup

2. **Velocity differences:** Corner kicks start from static positions,
   counterattacks involve fast transitions

3. **Spatial distribution:** Corner kick positions concentrated near goal,
   counterattack positions more distributed

**Recommendation:** Proceed with transfer learning experiments despite distribution
differences. The CrystalConv layers may still capture useful spatial-relational
patterns. Document performance degradation attributable to distribution shift.
""")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Transform DFL corners to USSF format')
    parser.add_argument('--adj-type', type=str, default='dense',
                        choices=['dense', 'normal'],
                        help='Adjacency type to use')
    parser.add_argument('--frame-selection', type=str, default='delivery',
                        choices=['delivery', 'sample', 'all'],
                        help='Which frames to include')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for transformed data')

    args = parser.parse_args()

    # Setup directories
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Engineer DFL Corner Features to Match USSF Schema")
    print("=" * 60)
    print(f"Adjacency type: {args.adj_type}")
    print(f"Frame selection: {args.frame_selection}")

    # Load DFL corner dataset
    print("\n--- Loading DFL Corner Dataset ---")
    with open(CORNER_DATASET, 'rb') as f:
        dfl_dataset = pickle.load(f)
    print(f"Loaded {len(dfl_dataset)} corners")

    # Load USSF data for comparison
    print("\n--- Loading USSF Data for Comparison ---")
    with open(USSF_DATA, 'rb') as f:
        ussf_data = pickle.load(f)
    print(f"Loaded USSF data with {len(ussf_data[args.adj_type]['x'])} graphs")

    # Transform DFL dataset
    print(f"\n--- Transforming DFL Corners ({args.adj_type} adjacency) ---")
    transformed_dataset = transform_corner_dataset(
        dfl_dataset,
        adj_type=args.adj_type,
        frame_selection=args.frame_selection
    )
    print(f"Transformed {len(transformed_dataset)} corners")

    # Compute statistics
    print("\n--- Computing Feature Statistics ---")
    dfl_stats = compute_feature_statistics(transformed_dataset)
    print(f"Total nodes: {dfl_stats['total_nodes']}")
    print(f"Total edges: {dfl_stats['total_edges']}")

    # Run KS tests
    print("\n--- Running KS Tests ---")
    ks_results = run_ks_tests(transformed_dataset, ussf_data, args.adj_type)

    n_node_mismatch = len(ks_results['summary']['significant_node_mismatches'])
    n_edge_mismatch = len(ks_results['summary']['significant_edge_mismatches'])
    print(f"Significant node mismatches: {n_node_mismatch}/12")
    print(f"Significant edge mismatches: {n_edge_mismatch}/6")

    if ks_results['summary']['significant_node_mismatches']:
        print(f"  Node: {', '.join(ks_results['summary']['significant_node_mismatches'])}")
    if ks_results['summary']['significant_edge_mismatches']:
        print(f"  Edge: {', '.join(ks_results['summary']['significant_edge_mismatches'])}")

    # Generate report
    print("\n--- Generating Report ---")
    report = generate_report(dfl_stats, ks_results, args.adj_type)
    report_path = REPORTS_DIR / f"dfl_ussf_distribution_comparison_{args.adj_type}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Save transformed dataset
    output_path = output_dir / f"dfl_corners_ussf_format_{args.adj_type}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(transformed_dataset, f)
    print(f"Transformed dataset saved to: {output_path}")

    # Save KS results
    ks_path = REPORTS_DIR / f"ks_test_results_{args.adj_type}.pkl"
    with open(ks_path, 'wb') as f:
        pickle.dump(ks_results, f)
    print(f"KS test results saved to: {ks_path}")

    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print(f"\nDeliverables:")
    print(f"  1. Transformed dataset: {output_path}")
    print(f"  2. Distribution report: {report_path}")
    print(f"  3. KS test results: {ks_path}")


if __name__ == "__main__":
    main()
