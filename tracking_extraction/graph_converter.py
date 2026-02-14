"""Convert unified CornerTrackingData to USSF-schema PyTorch Geometric graphs.

Produces graphs with 12 node features and 6 edge features matching the
schema used by the USSF pretrained backbone (transfer_learning/).

Node features (12):
    0. x                    - [0, 1] normalized pitch position
    1. y                    - [0, 1] normalized pitch position
    2. vx                   - Unit vector x component
    3. vy                   - Unit vector y component
    4. velocity_mag         - [0, 1] normalized speed
    5. velocity_angle       - [0, 1] normalized angle
    6. dist_goal            - [0, 1] normalized distance to goal
    7. angle_goal           - [0, 1] normalized angle to goal
    8. dist_ball            - [0, 1] normalized distance to ball
    9. angle_ball           - [0, 1] normalized angle to ball
    10. attacking_team_flag - 1.0 attacking, 0.0 defending, 0.0 ball
    11. potential_receiver  - Always 0.0 (USSF-specific, not used for corners)

Edge features (6):
    0. player_distance       - Normalized Euclidean distance
    1. speed_difference      - Signed speed difference [-1, 1]
    2. positional_sine_angle - (sin(angle) + 1) / 2
    3. positional_cosine_angle - (cos(angle) + 1) / 2
    4. velocity_sine_angle   - (sin(angle) + 1) / 2
    5. velocity_cosine_angle - (cos(angle) + 1) / 2
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch_geometric.data import Data

from .core import CornerTrackingData, Frame, PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)

MAX_VELOCITY = 10.0  # m/s normalization cap
GOAL_X = PITCH_LENGTH  # 105.0m — goal at far end
GOAL_Y = PITCH_WIDTH / 2.0  # 34.0m — center of goal


# --- Node Features ---

def _normalize_angle(angle_rad: float) -> float:
    """Normalize angle from [-pi, pi] to [0, 1]."""
    return (angle_rad + math.pi) / (2 * math.pi)


def _unit_vector(vx: float, vy: float):
    """Convert velocity to (unit_vx, unit_vy, magnitude)."""
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-8:
        return 1.0, 0.0, 0.0
    return vx / mag, vy / mag, mag


def _compute_node_features(
    x_m: float, y_m: float,
    vx: float, vy: float,
    team: str,
    is_ball: bool,
    ball_x_norm: float, ball_y_norm: float,
) -> np.ndarray:
    """Compute 12 USSF-schema node features for one player or the ball.

    Coordinates are normalized to [0, 1] using standard pitch dimensions.
    Goal is at (1.0, 0.5) in normalized coords.
    """
    # Normalize position to [0, 1]
    x_norm = np.clip(x_m / PITCH_LENGTH, 0.0, 1.0)
    y_norm = np.clip(y_m / PITCH_WIDTH, 0.0, 1.0)

    # Goal in normalized coords
    goal_x_norm = 1.0
    goal_y_norm = 0.5

    # Velocity: unit vector + normalized magnitude
    vx_unit, vy_unit, vel_mag = _unit_vector(vx, vy)
    vel_mag_norm = np.clip(vel_mag / MAX_VELOCITY, 0.0, 1.0)

    # Velocity angle
    vel_angle = math.atan2(vy_unit, vx_unit)
    vel_angle_norm = _normalize_angle(vel_angle)

    # Distance and angle to goal (in normalized coords)
    dx_goal = goal_x_norm - x_norm
    dy_goal = goal_y_norm - y_norm
    dist_goal = math.sqrt(dx_goal ** 2 + dy_goal ** 2)
    dist_goal_norm = np.clip(dist_goal / math.sqrt(2), 0.0, 1.0)
    angle_goal = math.atan2(dy_goal, dx_goal)
    angle_goal_norm = _normalize_angle(angle_goal)

    # Distance and angle to ball (in normalized coords)
    dx_ball = ball_x_norm - x_norm
    dy_ball = ball_y_norm - y_norm
    dist_ball = math.sqrt(dx_ball ** 2 + dy_ball ** 2)
    dist_ball_norm = np.clip(dist_ball / math.sqrt(2), 0.0, 1.0)
    angle_ball = math.atan2(dy_ball, dx_ball)
    angle_ball_norm = _normalize_angle(angle_ball)

    # Team flag
    if is_ball:
        attacking_flag = 0.0
    elif team == "attacking":
        attacking_flag = 1.0
    elif team == "defending":
        attacking_flag = 0.0
    else:
        attacking_flag = 0.5  # unknown

    return np.array([
        x_norm,             # 0
        y_norm,             # 1
        vx_unit,            # 2
        vy_unit,            # 3
        vel_mag_norm,       # 4
        vel_angle_norm,     # 5
        dist_goal_norm,     # 6
        angle_goal_norm,    # 7
        dist_ball_norm,     # 8
        angle_ball_norm,    # 9
        attacking_flag,     # 10
        0.0,                # 11 potential_receiver (always 0)
    ], dtype=np.float32)


# --- Edge Features ---

def _compute_edge_features(node_i: np.ndarray, node_j: np.ndarray) -> np.ndarray:
    """Compute 6 USSF-schema edge features between two nodes."""
    xi, yi = node_i[0], node_i[1]
    xj, yj = node_j[0], node_j[1]
    vxi, vyi = node_i[2], node_i[3]  # unit vectors
    vxj, vyj = node_j[2], node_j[3]
    vel_mag_i = node_i[4]
    vel_mag_j = node_j[4]

    # 1. Player distance
    dx = xj - xi
    dy = yj - yi
    dist = math.sqrt(dx * dx + dy * dy)
    dist_norm = np.clip(dist / math.sqrt(2), 0.0, 1.0)

    # 2. Speed difference (signed, in [-1, 1] range)
    speed_diff = vel_mag_j - vel_mag_i

    # 3-4. Positional angle features
    pos_angle = math.atan2(dy, dx)
    pos_sine = (math.sin(pos_angle) + 1) / 2
    pos_cosine = (math.cos(pos_angle) + 1) / 2

    # 5-6. Velocity angle features (angle between velocity vectors)
    dot = vxi * vxj + vyi * vyj
    cross = vxi * vyj - vyi * vxj
    vel_angle = math.atan2(cross, dot)
    vel_sine = (math.sin(vel_angle) + 1) / 2
    vel_cosine = (math.cos(vel_angle) + 1) / 2

    return np.array([
        dist_norm,    # 0
        speed_diff,   # 1
        pos_sine,     # 2
        pos_cosine,   # 3
        vel_sine,     # 4
        vel_cosine,   # 5
    ], dtype=np.float32)


# --- Direction Normalization ---

def _detect_attack_direction(frame: Frame) -> str:
    """Detect which end the attacking team is targeting from player positions.

    Uses the defending goalkeeper position as the most reliable signal:
    the defending GK stands near the goal being attacked.

    Returns:
        "right" if attacked goal is at x=105 (no flip needed)
        "left" if attacked goal is at x=0 (flip needed)
    """
    # Find defending goalkeeper
    for pf in frame.players:
        if pf.team == "defending" and pf.role == "goalkeeper":
            return "right" if pf.x > PITCH_LENGTH / 2 else "left"

    # Fallback: use mean defending player position
    def_xs = [pf.x for pf in frame.players if pf.team == "defending"]
    if def_xs:
        return "right" if (sum(def_xs) / len(def_xs)) > PITCH_LENGTH / 2 else "left"

    # Last resort: assume attacking toward x=105
    return "right"


# --- Graph Construction ---

def corner_to_ussf_graph(
    corner: CornerTrackingData,
    adjacency: str = "dense",
) -> Optional[Data]:
    """Convert a single corner to a USSF-schema PyG Data object.

    Uses the delivery frame snapshot to build one graph per corner.
    Normalizes attack direction so the attacked goal is always at x=1.0.

    Args:
        corner: Unified CornerTrackingData
        adjacency: "dense" (fully connected) or "normal" (kNN + team)

    Returns:
        PyG Data object or None if delivery frame is invalid
    """
    if not corner.frames:
        logger.warning("%s: no frames", corner.corner_id)
        return None

    # Get delivery frame
    d_idx = corner.delivery_frame
    if d_idx < 0 or d_idx >= len(corner.frames):
        logger.warning("%s: delivery_frame=%d out of range [0, %d)",
                       corner.corner_id, d_idx, len(corner.frames))
        d_idx = len(corner.frames) // 2  # fallback: middle frame

    frame = corner.frames[d_idx]

    if not frame.players:
        logger.warning("%s: delivery frame has no players", corner.corner_id)
        return None

    # Detect attack direction and determine if x-flip is needed
    direction = _detect_attack_direction(frame)
    flip_x = (direction == "left")  # flip when attacking toward x=0

    # Ball position in meters, then normalize
    ball_x_m = frame.ball_x if frame.ball_x is not None else PITCH_LENGTH
    ball_y_m = frame.ball_y if frame.ball_y is not None else 0.0
    if flip_x:
        ball_x_m = PITCH_LENGTH - ball_x_m
    ball_x_norm = np.clip(ball_x_m / PITCH_LENGTH, 0.0, 1.0)
    ball_y_norm = np.clip(ball_y_m / PITCH_WIDTH, 0.0, 1.0)

    # Build node features for all players
    node_features = []
    positions = []
    for pf in frame.players:
        vx = pf.vx if pf.vx is not None else 0.0
        vy = pf.vy if pf.vy is not None else 0.0
        px = pf.x
        if flip_x:
            px = PITCH_LENGTH - px
            vx = -vx
        feat = _compute_node_features(
            px, pf.y, vx, vy,
            team=pf.team,
            is_ball=False,
            ball_x_norm=ball_x_norm,
            ball_y_norm=ball_y_norm,
        )
        node_features.append(feat)
        positions.append([np.clip(px / PITCH_LENGTH, 0.0, 1.0),
                          np.clip(pf.y / PITCH_WIDTH, 0.0, 1.0)])

    # Add ball node
    ball_feat = _compute_node_features(
        ball_x_m, ball_y_m, 0.0, 0.0,
        team="ball",
        is_ball=True,
        ball_x_norm=ball_x_norm,
        ball_y_norm=ball_y_norm,
    )
    node_features.append(ball_feat)
    positions.append([ball_x_norm, ball_y_norm])

    nodes = np.array(node_features, dtype=np.float32)
    n_nodes = len(nodes)

    # Build edges
    if adjacency == "dense":
        edge_index = _build_dense_adjacency(n_nodes)
    elif adjacency == "normal":
        edge_index = _build_normal_adjacency(nodes)
    else:
        raise ValueError(f"Unknown adjacency type: {adjacency}")

    # Compute edge features
    n_edges = edge_index.shape[1]
    edge_attrs = np.zeros((n_edges, 6), dtype=np.float32)
    for e in range(n_edges):
        i = edge_index[0, e].item()
        j = edge_index[1, e].item()
        edge_attrs[e] = _compute_edge_features(nodes[i], nodes[j])

    x_tensor = torch.tensor(nodes, dtype=torch.float32)
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr_tensor,
        pos=pos_tensor,
    )


def _build_dense_adjacency(n_nodes: int) -> torch.Tensor:
    """Fully connected directed edges (i != j)."""
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _build_normal_adjacency(nodes: np.ndarray, k_neighbors: int = 4) -> torch.Tensor:
    """kNN + team-based + ball connectivity."""
    from scipy.spatial.distance import cdist

    n_nodes = len(nodes)
    edge_set = set()

    # Separate players and ball
    player_idx = []
    ball_idx = None
    for i in range(n_nodes):
        if nodes[i, 8] < 0.01:  # dist_ball ~ 0 → ball node
            ball_idx = i
        else:
            player_idx.append(i)

    if not player_idx:
        return torch.empty((2, 0), dtype=torch.long)

    n_players = len(player_idx)

    # kNN edges
    if n_players > k_neighbors:
        player_pos = nodes[player_idx, :2]
        dist_matrix = cdist(player_pos, player_pos)
        np.fill_diagonal(dist_matrix, np.inf)
        for i, pi in enumerate(player_idx):
            neighbors = np.argsort(dist_matrix[i])[:k_neighbors]
            for j in neighbors:
                pj = player_idx[j]
                edge_set.add((pi, pj))
    else:
        for pi in player_idx:
            for pj in player_idx:
                if pi != pj:
                    edge_set.add((pi, pj))

    # Ball edges
    if ball_idx is not None:
        for pi in player_idx:
            edge_set.add((pi, ball_idx))
            edge_set.add((ball_idx, pi))

    # Team connectivity
    attackers = [i for i in player_idx if nodes[i, 10] > 0.5]
    defenders = [i for i in player_idx if nodes[i, 10] < 0.5]
    if attackers and defenders:
        att_pos = nodes[attackers, :2]
        def_pos = nodes[defenders, :2]
        cross_dist = cdist(att_pos, def_pos)
        for i, att_i in enumerate(attackers):
            nearest = defenders[np.argmin(cross_dist[i])]
            edge_set.add((att_i, nearest))
            edge_set.add((nearest, att_i))

    if edge_set:
        edges = list(edge_set)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        return torch.tensor([src, dst], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


# --- Dataset Conversion ---

def convert_dataset(
    corners: List[CornerTrackingData],
    adjacency: str = "dense",
) -> List[Dict[str, Any]]:
    """Convert a list of corners to USSF-schema graph dataset.

    Output format matches transfer_learning/data/dfl_corners_ussf_format_dense.pkl:
    each entry is {'graphs': [Data], 'labels': dict, 'match_id': str, 'corner_time': float}

    Args:
        corners: List of CornerTrackingData
        adjacency: "dense" or "normal"

    Returns:
        List of dicts, one per corner
    """
    dataset = []
    for corner in corners:
        graph = corner_to_ussf_graph(corner, adjacency=adjacency)
        if graph is None:
            logger.warning("Skipping %s: graph conversion failed", corner.corner_id)
            continue

        # Map outcome to binary labels
        shot_binary = 1 if corner.outcome == "shot" else 0
        labels = {
            "shot_binary": shot_binary,
            "goal_binary": 0,  # unified format doesn't distinguish goals
            "first_contact_team": "unknown",
            "outcome_class": "shot_saved" if shot_binary else "other",
        }

        # Extract corner time from metadata
        corner_time = corner.metadata.get("corner_time_s", 0.0)
        if corner_time == 0.0:
            # Estimate from delivery frame
            corner_time = (corner.delivery_frame / corner.fps) if corner.fps > 0 else 0.0

        dataset.append({
            "graphs": [graph],
            "labels": labels,
            "match_id": corner.match_id,
            "corner_time": corner_time,
            "source": corner.source,
            "corner_id": corner.corner_id,
        })

    logger.info("Converted %d / %d corners to graphs", len(dataset), len(corners))
    return dataset


def save_graph_dataset(dataset: List[Dict[str, Any]], path: Path) -> None:
    """Save graph dataset as pickle (same format as transfer_learning/data/)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    logger.info("Saved %d graphs to %s", len(dataset), path)


def load_graph_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load graph dataset from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Train/Val/Test Splits ---

def create_splits(
    dataset: List[Dict[str, Any]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create match-based train/val/test splits with shot-rate stratification.

    All corners from the same match go to the same split.
    Matches are sorted by shot rate and distributed round-robin across
    splits so each split gets a mix of high and low shot-rate matches.

    Args:
        dataset: Full graph dataset
        val_fraction: Target fraction for validation
        test_fraction: Target fraction for test
        seed: Random seed

    Returns:
        Dict with keys "train", "val", "test", each a list of dataset entries
    """
    rng = np.random.RandomState(seed)

    # Group by match
    match_to_entries: Dict[str, List[Dict[str, Any]]] = {}
    for entry in dataset:
        mid = entry["match_id"]
        match_to_entries.setdefault(mid, []).append(entry)

    # Compute shot rate per match
    match_ids = list(match_to_entries.keys())
    match_sizes = {mid: len(entries) for mid, entries in match_to_entries.items()}
    match_shot_rates = {}
    for mid, entries in match_to_entries.items():
        shots = sum(1 for e in entries if e["labels"]["shot_binary"] == 1)
        match_shot_rates[mid] = shots / len(entries) if entries else 0

    # Sort by shot rate for stratified distribution, then shuffle within
    # equal-rate groups for randomness
    rng.shuffle(match_ids)  # shuffle first so ties are random
    match_ids.sort(key=lambda mid: match_shot_rates[mid])

    total = len(dataset)
    target_test = int(total * test_fraction)
    target_val = int(total * val_fraction)

    # Distribute matches round-robin: walk sorted list, assign to the
    # split that is furthest below its target count. This ensures each
    # split gets a mix of high and low shot-rate matches.
    target_train = total - target_test - target_val
    split_counts = {"test": 0, "val": 0, "train": 0}
    split_targets = {"test": target_test, "val": target_val, "train": target_train}
    split_matches: Dict[str, List[str]] = {"test": [], "val": [], "train": []}

    # Assign matches using interleaved blocks to ensure stratification.
    # Group matches into 3-match blocks (sorted by shot rate); within each
    # block, assign one match to each split (picking the most-deficit split).
    # This ensures each split gets a balanced mix of shot rates.
    for i, mid in enumerate(match_ids):
        n = match_sizes[mid]
        # Pick the split furthest below its target fraction
        best_split = None
        best_deficit = -float("inf")
        for s in ["test", "val", "train"]:
            fraction = split_counts[s] / max(1, sum(split_counts.values()))
            target_frac = split_targets[s] / total
            deficit = target_frac - fraction
            if deficit > best_deficit:
                best_deficit = deficit
                best_split = s
        split_matches[best_split].append(mid)
        split_counts[best_split] += n

    splits = {
        s: [e for mid in split_matches[s] for e in match_to_entries[mid]]
        for s in ["train", "val", "test"]
    }

    # Log split info
    for split_name, entries in splits.items():
        n = len(entries)
        shots = sum(1 for e in entries if e["labels"]["shot_binary"] == 1)
        matches = set(e["match_id"] for e in entries)
        logger.info("  %s: %d corners (%d shots, %.0f%%) from %d matches",
                     split_name, n, shots, 100 * shots / n if n else 0, len(matches))

    return splits


def print_graph_summary(dataset: List[Dict[str, Any]]) -> None:
    """Print summary statistics for a graph dataset."""
    if not dataset:
        print("Empty dataset")
        return

    n = len(dataset)
    shots = sum(1 for d in dataset if d["labels"]["shot_binary"] == 1)
    matches = set(d["match_id"] for d in dataset)

    # Graph statistics
    n_nodes_list = [d["graphs"][0].x.shape[0] for d in dataset]
    n_edges_list = [d["graphs"][0].edge_index.shape[1] for d in dataset]

    # Source breakdown (infer from match_id prefix or corner_id convention)
    sources = {}
    for d in dataset:
        mid = d["match_id"]
        if mid.startswith("DFL"):
            src = "dfl"
        else:
            try:
                int(mid)
                src = "skillcorner"
            except ValueError:
                src = "unknown"
        sources[src] = sources.get(src, 0) + 1

    print(f"\n{'='*60}")
    print(f"Graph Dataset Summary: {n} corners")
    print(f"{'='*60}")
    print(f"Shots: {shots} ({100*shots/n:.1f}%), No-shot: {n-shots} ({100*(n-shots)/n:.1f}%)")
    print(f"Matches: {len(matches)}")
    print(f"\nBy source:")
    for src, cnt in sorted(sources.items()):
        print(f"  {src}: {cnt}")
    print(f"\nGraph size:")
    print(f"  Nodes: min={min(n_nodes_list)}, max={max(n_nodes_list)}, "
          f"mean={sum(n_nodes_list)/n:.1f}")
    print(f"  Edges: min={min(n_edges_list)}, max={max(n_edges_list)}, "
          f"mean={sum(n_edges_list)/n:.1f}")

    # Feature range check
    all_x = torch.cat([d["graphs"][0].x for d in dataset], dim=0)
    print(f"\nNode features (12):")
    names = ["x", "y", "vx_unit", "vy_unit", "vel_mag", "vel_angle",
             "dist_goal", "angle_goal", "dist_ball", "angle_ball",
             "atk_flag", "pot_recv"]
    for i, name in enumerate(names):
        col = all_x[:, i]
        print(f"  {name:12s}: min={col.min():.3f}, max={col.max():.3f}, "
              f"mean={col.mean():.3f}, std={col.std():.3f}")

    if dataset[0]["graphs"][0].edge_attr is not None:
        all_ea = torch.cat([d["graphs"][0].edge_attr for d in dataset], dim=0)
        print(f"\nEdge features (6):")
        enames = ["distance", "speed_diff", "pos_sin", "pos_cos", "vel_sin", "vel_cos"]
        for i, name in enumerate(enames):
            col = all_ea[:, i]
            print(f"  {name:12s}: min={col.min():.3f}, max={col.max():.3f}, "
                  f"mean={col.mean():.3f}, std={col.std():.3f}")

    print(f"{'='*60}\n")
