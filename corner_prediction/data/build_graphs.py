"""Build PyTorch Geometric graphs from extracted corner kick records.

Converts the output of extract_corners.py (86 corner records with 22 players
each) into PyG Data objects suitable for GNN training.

Node features (13 per player):
    0. x              - Normalized x position (x / 52.5, range [-1, 1])
    1. y              - Normalized y position (y / 34.0, range [-1, 1])
    2. vx             - x velocity in m/s
    3. vy             - y velocity in m/s
    4. speed          - Scalar speed in m/s
    5. is_attacking   - 1.0 if attacking team, 0.0 if defending
    6. is_corner_taker - 1.0 if corner taker
    7. is_goalkeeper  - 1.0 if goalkeeper
    8. is_detected    - 1.0 if detected (not extrapolated)
    9. group_GK       - Position group one-hot
    10. group_DEF
    11. group_MID
    12. group_FWD

Edge features (4):
    0. dx             - x difference (normalized coords)
    1. dy             - y difference (normalized coords)
    2. distance       - Euclidean distance (normalized coords)
    3. same_team      - 1.0 if same team, 0.0 if different

Usage:
    python -m corner_prediction.data.build_graphs
    python -m corner_prediction.data.build_graphs --edge-type dense
    python -m corner_prediction.data.build_graphs --input path/to/records.pkl --output path/to/graphs.pkl
"""

import argparse
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_FEATURE_DIM = 13
EDGE_FEATURE_DIM = 4

HALF_LENGTH = 52.5  # meters (pitch half-length)
HALF_WIDTH = 34.0   # meters (pitch half-width)

# Coarse position groups for one-hot encoding
POSITION_GROUPS = ["GK", "DEF", "MID", "FWD"]

# Map from role abbreviation (in extracted records) to coarse group
ROLE_TO_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "DM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FWD", "RW": "FWD", "LF": "FWD", "RF": "FWD", "CF": "FWD",
    "SUB": "MID",
}


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def build_node_features(player: Dict[str, Any]) -> List[float]:
    """Build 13-dim feature vector for a single player.

    Args:
        player: Player dict from extracted corner record.

    Returns:
        List of 13 floats.
    """
    x_norm = player["x"] / HALF_LENGTH
    y_norm = player["y"] / HALF_WIDTH

    vx = player.get("vx", 0.0)
    vy = player.get("vy", 0.0)
    speed = player.get("speed", 0.0)

    is_attacking = 1.0 if player["is_attacking"] else 0.0
    is_corner_taker = 1.0 if player["is_corner_taker"] else 0.0
    is_goalkeeper = 1.0 if player["is_goalkeeper"] else 0.0
    is_detected = 1.0 if player["is_detected"] else 0.0

    # Position group one-hot
    role = player.get("role", "SUB")
    group = ROLE_TO_GROUP.get(role, "MID")
    group_onehot = [1.0 if g == group else 0.0 for g in POSITION_GROUPS]

    return [
        x_norm,          # 0
        y_norm,          # 1
        vx,              # 2
        vy,              # 3
        speed,           # 4
        is_attacking,    # 5
        is_corner_taker, # 6
        is_goalkeeper,   # 7
        is_detected,     # 8
    ] + group_onehot     # 9-12


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def build_edge_features(
    xi: float, yi: float, team_i: float,
    xj: float, yj: float, team_j: float,
) -> List[float]:
    """Build 4-dim edge feature vector between two nodes.

    All positions are in normalized coordinates (x/HALF_LENGTH, y/HALF_WIDTH).

    Args:
        xi, yi: Node i position (normalized).
        team_i: 1.0 if attacking, 0.0 if defending.
        xj, yj: Node j position (normalized).
        team_j: 1.0 if attacking, 0.0 if defending.

    Returns:
        List of 4 floats: [dx, dy, distance, same_team].
    """
    dx = xj - xi
    dy = yj - yi
    distance = math.sqrt(dx * dx + dy * dy)
    same_team = 1.0 if team_i == team_j else 0.0

    return [dx, dy, distance, same_team]


def build_knn_edges(positions: np.ndarray, k: int = 6) -> torch.Tensor:
    """Build k-nearest-neighbor edge index.

    Args:
        positions: [n_nodes, 2] array of (x, y) positions.
        k: Number of nearest neighbors per node.

    Returns:
        edge_index: [2, n_nodes * k] tensor (or dense if n_nodes <= k).
    """
    from scipy.spatial.distance import cdist

    n_nodes = positions.shape[0]

    # Degenerate case: fewer nodes than k+1 → fully connected
    if n_nodes <= k:
        return build_dense_edges(n_nodes)

    dist_matrix = cdist(positions, positions)
    np.fill_diagonal(dist_matrix, np.inf)

    src, dst = [], []
    for i in range(n_nodes):
        neighbors = np.argsort(dist_matrix[i])[:k]
        for j in neighbors:
            src.append(i)
            dst.append(int(j))

    return torch.tensor([src, dst], dtype=torch.long)


def build_dense_edges(n_nodes: int) -> torch.Tensor:
    """Build fully connected edge index (all pairs i != j).

    Args:
        n_nodes: Number of nodes.

    Returns:
        edge_index: [2, n_nodes * (n_nodes - 1)] tensor.
    """
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def corner_record_to_graph(
    record: Dict[str, Any],
    edge_type: str = "knn",
    k: int = 6,
) -> Data:
    """Convert a single corner record to a PyG Data object.

    Args:
        record: Corner record dict from extract_corners.py.
        edge_type: "knn" or "dense".
        k: Number of neighbors for KNN edges.

    Returns:
        PyG Data object with 22 nodes and all labels/metadata.
    """
    players = record["players"]
    n_nodes = len(players)

    # Build node features [n_nodes, 13]
    node_feats = [build_node_features(p) for p in players]
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Extract normalized positions for edge construction
    positions = np.array(
        [[p["x"] / HALF_LENGTH, p["y"] / HALF_WIDTH] for p in players],
        dtype=np.float64,
    )

    # Build edges
    if edge_type == "knn":
        edge_index = build_knn_edges(positions, k=k)
    elif edge_type == "dense":
        edge_index = build_dense_edges(n_nodes)
    else:
        raise ValueError(f"Unknown edge_type: {edge_type!r}")

    # Build edge features [n_edges, 4]
    n_edges = edge_index.shape[1]
    edge_feats = []
    for e in range(n_edges):
        i = edge_index[0, e].item()
        j = edge_index[1, e].item()
        edge_feats.append(build_edge_features(
            positions[i, 0], positions[i, 1], x[i, 5].item(),  # xi, yi, team_i
            positions[j, 0], positions[j, 1], x[j, 5].item(),  # xj, yj, team_j
        ))
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    # Receiver labels (Stage 1)
    # Mask: attacking outfield players only (valid receiver candidates)
    receiver_mask = torch.tensor(
        [p["is_attacking"] and not p["is_goalkeeper"] for p in players],
        dtype=torch.bool,
    )
    receiver_label = torch.tensor(
        [1.0 if p["is_receiver"] else 0.0 for p in players],
        dtype=torch.float32,
    )
    has_receiver_label = record["has_receiver_label"]

    # Validate: receiver must be within the mask (attacking outfield).
    # If the first contact was by a defender or GK, this corner has no
    # valid Stage 1 label — zero out and mark as unlabeled.
    if has_receiver_label and receiver_label.sum() > 0:
        recv_idx = receiver_label.argmax().item()
        if not receiver_mask[recv_idx]:
            has_receiver_label = False
            receiver_label = torch.zeros(n_nodes, dtype=torch.float32)

    # Shot labels (Stage 2)
    shot_label = 1 if record["lead_to_shot"] else 0
    goal_label = 1 if record["lead_to_goal"] else 0

    # Graph-level features
    corner_side = 1.0 if record.get("corner_side") == "right" else 0.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,

        # Stage 1 labels
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=has_receiver_label,

        # Stage 2 labels
        shot_label=shot_label,
        goal_label=goal_label,

        # Graph-level features
        corner_side=corner_side,

        # Metadata
        match_id=str(record["match_id"]),
        corner_id=record["corner_id"],
        detection_rate=record["detection_rate"],
        source=record.get("source", "skillcorner"),
    )


def build_graph_dataset(
    records: List[Dict[str, Any]],
    edge_type: str = "knn",
    k: int = 6,
) -> List[Data]:
    """Convert all corner records to PyG Data objects.

    Args:
        records: List of corner record dicts.
        edge_type: "knn" or "dense".
        k: Number of neighbors for KNN edges.

    Returns:
        List of PyG Data objects.
    """
    graphs = []
    for record in records:
        try:
            graph = corner_record_to_graph(record, edge_type=edge_type, k=k)
            graphs.append(graph)
        except Exception:
            logger.exception("Failed to convert corner %s", record.get("corner_id", "?"))
    logger.info("Built %d / %d graphs (edge_type=%s, k=%d)",
                len(graphs), len(records), edge_type, k)
    return graphs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(graphs: List[Data]) -> None:
    """Print summary statistics for a graph dataset."""
    if not graphs:
        print("Empty dataset")
        return

    n = len(graphs)
    n_shots = sum(1 for g in graphs if g.shot_label == 1)
    n_goals = sum(1 for g in graphs if g.goal_label == 1)
    n_receiver = sum(1 for g in graphs if g.has_receiver_label)
    matches = sorted(set(g.match_id for g in graphs))

    n_nodes_list = [g.x.shape[0] for g in graphs]
    n_edges_list = [g.edge_index.shape[1] for g in graphs]

    print(f"\n{'=' * 60}")
    print(f"Graph Dataset Summary: {n} graphs")
    print(f"{'=' * 60}")
    print(f"Matches:         {len(matches)}")
    print(f"Shots:           {n_shots}/{n} ({100 * n_shots / n:.1f}%)")
    print(f"Goals:           {n_goals}/{n} ({100 * n_goals / n:.1f}%)")
    print(f"Receiver labels: {n_receiver}/{n} ({100 * n_receiver / n:.1f}%)")

    print(f"\nGraph structure:")
    print(f"  Nodes: min={min(n_nodes_list)}, max={max(n_nodes_list)}, "
          f"mean={sum(n_nodes_list) / n:.1f}")
    print(f"  Edges: min={min(n_edges_list)}, max={max(n_edges_list)}, "
          f"mean={sum(n_edges_list) / n:.1f}")

    # Node feature statistics
    all_x = torch.cat([g.x for g in graphs], dim=0)
    feat_names = [
        "x_norm", "y_norm", "vx", "vy", "speed",
        "is_atk", "is_taker", "is_gk", "is_det",
        "grp_GK", "grp_DEF", "grp_MID", "grp_FWD",
    ]
    print(f"\nNode features ({all_x.shape[1]}):")
    for i, name in enumerate(feat_names):
        col = all_x[:, i]
        print(f"  {name:10s}: min={col.min().item():7.3f}, max={col.max().item():7.3f}, "
              f"mean={col.mean().item():7.3f}, std={col.std().item():7.3f}")

    # Edge feature statistics
    all_ea = torch.cat([g.edge_attr for g in graphs], dim=0)
    edge_names = ["dx", "dy", "distance", "same_team"]
    print(f"\nEdge features ({all_ea.shape[1]}):")
    for i, name in enumerate(edge_names):
        col = all_ea[:, i]
        print(f"  {name:10s}: min={col.min().item():7.3f}, max={col.max().item():7.3f}, "
              f"mean={col.mean().item():7.3f}, std={col.std().item():7.3f}")

    # Receiver mask stats
    total_candidates = sum(g.receiver_mask.sum().item() for g in graphs)
    print(f"\nReceiver candidates per graph: "
          f"mean={total_candidates / n:.1f}")

    # Detection rate
    det_rates = [g.detection_rate for g in graphs]
    print(f"Detection rate: min={min(det_rates):.2f}, max={max(det_rates):.2f}, "
          f"mean={sum(det_rates) / n:.2f}")

    # Source distribution
    sources = {}
    for g in graphs:
        src = getattr(g, "source", "skillcorner")
        sources[src] = sources.get(src, 0) + 1
    if len(sources) > 1:
        print(f"\nSource distribution:")
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count} ({100 * count / n:.1f}%)")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build PyG graphs from extracted corner records",
    )
    parser.add_argument(
        "--input", type=str,
        default="corner_prediction/data/extracted_corners.pkl",
        help="Path to extracted corner records (pickle)",
    )
    parser.add_argument(
        "--output", type=str,
        default="corner_prediction/data/graphs.pkl",
        help="Output path for graph dataset (pickle)",
    )
    parser.add_argument(
        "--edge-type", type=str, default="knn",
        choices=["knn", "dense"],
        help="Edge construction method",
    )
    parser.add_argument("--k", type=int, default=6, help="KNN neighbor count")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load records
    input_path = Path(args.input)
    with open(input_path, "rb") as f:
        records = pickle.load(f)
    logger.info("Loaded %d corner records from %s", len(records), input_path)

    # Build graphs
    graphs = build_graph_dataset(records, edge_type=args.edge_type, k=args.k)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graphs, f)
    logger.info("Saved %d graphs to %s", len(graphs), output_path)

    # Summary
    print_summary(graphs)


if __name__ == "__main__":
    main()
