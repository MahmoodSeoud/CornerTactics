#!/usr/bin/env python3
"""Option A: Run the USSF-aligned GNN pipeline on StatsBomb freeze-frame data.

Converts StatsBomb corner kick freeze-frames into the 12-feature USSF schema
(with velocity features = 0), builds dense graphs, and runs LOMO CV + permutation
test with the frozen USSF backbone. This isolates data modality as the variable:
same backbone, same features, same protocol — different data source.

StatsBomb freeze-frames provide only a single positional snapshot, so:
  - Node features 2-5 (vx_unit, vy_unit, vel_mag, vel_angle) = 0
  - Edge features 1, 4, 5 (speed_diff, vel_sin, vel_cos) = 0
  → Effectively the position-only condition, but on StatsBomb data.

USSF 12-feature schema mapping from StatsBomb:
  0: x          ← SB location[0] / 120.0  (pitch-normalized [0,1])
  1: y          ← SB location[1] / 80.0   (pitch-normalized [0,1])
  2: vx_unit    ← 0.0 (no velocity from single frame)
  3: vy_unit    ← 0.0
  4: vel_mag    ← 0.0
  5: vel_angle  ← 0.5 (atan2(0,0)+pi)/(2pi)
  6: dist_goal  ← Euclidean to (1.0, 0.5), normalized
  7: angle_goal ← atan2 to (1.0, 0.5), normalized
  8: dist_ball  ← Euclidean to ball (corner location), normalized
  9: angle_ball ← atan2 to ball, normalized
  10: attacking_team ← SB teammate flag
  11: potential_receiver ← 0.0 (set at forward time)

Usage:
    cd /home/mseo/CornerTactics
    source FAANTRA/venv/bin/activate
    python scripts/statsbomb_ussf_pipeline.py
    python scripts/statsbomb_ussf_pipeline.py --permutation
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corner_prediction.config import (
    DIST_BALL_NORM,
    DIST_GOAL_NORM,
    EDGE_DIST_NORM,
    GOAL_X,
    GOAL_Y,
    N_PERMUTATIONS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PRETRAINED_PATH,
    RESULTS_DIR,
)
from corner_prediction.data.build_graphs import (
    _SELF_LOOP_EDGE,
    build_ussf_dense_edges,
    build_ussf_edge_features_vec,
)
from corner_prediction.config import (
    BATCH_SIZE,
    RECEIVER_DROPOUT,
    RECEIVER_EPOCHS,
    RECEIVER_HIDDEN,
    RECEIVER_LR,
    RECEIVER_PATIENCE,
    RECEIVER_WEIGHT_DECAY,
    SHOT_DROPOUT,
    SHOT_EPOCHS,
    SHOT_HIDDEN,
    SHOT_LR,
    SHOT_PATIENCE,
    SHOT_POS_WEIGHT,
    SHOT_WEIGHT_DECAY,
)
from corner_prediction.training.evaluate import (
    compute_receiver_metrics,
    compute_shot_metrics,
    save_results,
)
from corner_prediction.training.train import (
    build_model,
    eval_receiver,
    eval_shot,
    train_fold,
)
from corner_prediction.training.permutation_test import shuffle_shot_labels

# StatsBomb pitch dimensions
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0

# Minimum players required in freeze-frame
MIN_PLAYERS = 15


def load_corner_events(data_dir: str = "data/statsbomb") -> list:
    """Load corner kick pass events from ALL event files (3464 matches).

    Scans every match event file in events/events/ for corner kick passes,
    then matches them to freeze-frame data by UUID. This captures all 1800+
    corners with 360 coverage, not just the 100 matches in master_event_sequence.

    Returns list of dicts with event_uuid, match_id, location, shot label,
    freeze_frame, and goal label.
    """
    events_dir = os.path.join(data_dir, "events", "events")
    ff_dir = os.path.join(data_dir, "freeze-frames")

    # Load freeze-frame UUID lookup
    print("  Loading freeze-frame files...")
    ff_lookup = {}
    for fname in os.listdir(ff_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ff_dir, fname)) as f:
            events = json.load(f)
        for evt in events:
            ff_lookup[evt["event_uuid"]] = evt["freeze_frame"]
    print(f"  Loaded {len(ff_lookup)} freeze-frame events from {len(os.listdir(ff_dir))} files")

    # Scan all match event files
    print(f"  Scanning {len(os.listdir(events_dir))} match event files...")
    results = []
    skipped_no_ff = 0

    for fname in sorted(os.listdir(events_dir)):
        if not fname.endswith(".json"):
            continue
        mid = fname.replace(".json", "")
        with open(os.path.join(events_dir, fname)) as f:
            events = json.load(f)

        # Index by possession for shot detection
        possession_events = {}
        for e in events:
            poss = e.get("possession", -1)
            possession_events.setdefault(poss, []).append(e)

        # Find corner kick passes
        for e in events:
            if (e.get("type", {}).get("name") != "Pass"
                    or e.get("pass", {}).get("type", {}).get("name") != "Corner"):
                continue

            uuid = e["id"]
            if uuid not in ff_lookup:
                skipped_no_ff += 1
                continue

            poss = e.get("possession", -1)
            same_poss = possession_events.get(poss, [])

            lead_to_shot = any(ev["type"]["name"] == "Shot" for ev in same_poss)
            lead_to_goal = any(
                ev["type"]["name"] == "Shot"
                and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                for ev in same_poss
            )

            loc = e.get("location", [120.0, 40.0])
            corner_side = "right" if loc[1] >= 40.0 else "left"

            results.append({
                "event_uuid": uuid,
                "match_id": mid,
                "location": loc,
                "lead_to_shot": lead_to_shot,
                "lead_to_goal": lead_to_goal,
                "corner_side": corner_side,
                "freeze_frame": ff_lookup[uuid],
            })

    print(f"  Skipped (no freeze-frame): {skipped_no_ff}")
    return results


def load_freeze_frames(data_dir: str = "data/statsbomb") -> dict:
    """Legacy — freeze-frames are now loaded inline by load_corner_events."""
    raise NotImplementedError("Use load_corner_events() which loads FF inline")


def sb_to_ussf_node_features(
    sb_x: float, sb_y: float,
    ball_x_norm: float, ball_y_norm: float,
    is_attacking: bool,
) -> list:
    """Convert StatsBomb player position to 12-dim USSF node features.

    StatsBomb coords: x in [0, 120], y in [0, 80].
    USSF coords: x, y in [0, 1].
    Velocity features are all zero (single frame).
    """
    # Normalize to [0, 1]
    px = sb_x / SB_PITCH_LENGTH
    py = sb_y / SB_PITCH_WIDTH

    # Distance/angle to goal (1.0, 0.5) in normalized coords
    dx_goal = (px - GOAL_X) * PITCH_LENGTH
    dy_goal = (py - GOAL_Y) * PITCH_WIDTH
    dist_goal = math.sqrt(dx_goal**2 + dy_goal**2) / DIST_GOAL_NORM
    angle_goal = (math.atan2(dy_goal, dx_goal) + math.pi) / (2.0 * math.pi)

    # Distance/angle to ball in normalized coords
    dx_ball = (px - ball_x_norm) * PITCH_LENGTH
    dy_ball = (py - ball_y_norm) * PITCH_WIDTH
    dist_ball = math.sqrt(dx_ball**2 + dy_ball**2) / DIST_BALL_NORM
    angle_ball = (math.atan2(dy_ball, dx_ball) + math.pi) / (2.0 * math.pi)

    return [
        px, py,          # 0-1: position [0,1]
        0.0, 0.0,        # 2-3: vx_unit, vy_unit (no velocity)
        0.0,             # 4: vel_mag
        0.5,             # 5: vel_angle = (atan2(0,0)+pi)/(2pi) = 0.5
        dist_goal,       # 6
        angle_goal,      # 7
        dist_ball,       # 8
        angle_ball,      # 9
        1.0 if is_attacking else 0.0,  # 10: attacking_team
        0.0,             # 11: potential_receiver (set at forward time)
    ]


def sb_to_ussf_ball_features(ball_x_norm: float, ball_y_norm: float) -> list:
    """Build USSF ball node features from StatsBomb corner location."""
    dx_goal = (ball_x_norm - GOAL_X) * PITCH_LENGTH
    dy_goal = (ball_y_norm - GOAL_Y) * PITCH_WIDTH
    dist_goal = math.sqrt(dx_goal**2 + dy_goal**2) / DIST_GOAL_NORM
    angle_goal = (math.atan2(dy_goal, dx_goal) + math.pi) / (2.0 * math.pi)

    return [
        ball_x_norm, ball_y_norm,
        0.0, 0.0,        # vx, vy
        0.0,             # vel_mag
        0.5,             # vel_angle
        dist_goal,
        angle_goal,
        0.0,             # dist_ball = 0 (self)
        0.0,             # angle_ball = 0 (convention)
        0.0,             # attacking_team (neutral)
        0.0,             # potential_receiver
    ]


def build_statsbomb_ussf_graph(
    corner_event: dict,
    freeze_frame: list = None,
) -> Data:
    """Convert a StatsBomb corner + freeze-frame to USSF-aligned PyG graph.

    Produces N+1 nodes (N players + 1 ball), 12 features, dense edges with
    self-loops.
    """
    if freeze_frame is None:
        freeze_frame = corner_event["freeze_frame"]
    ball_loc = corner_event["location"]
    ball_x_norm = ball_loc[0] / SB_PITCH_LENGTH
    ball_y_norm = ball_loc[1] / SB_PITCH_WIDTH

    # Build node features for each player
    node_feats = []
    for player in freeze_frame:
        loc = player["location"]
        is_attacking = player["teammate"]  # teammate = attacking team
        node_feats.append(sb_to_ussf_node_features(
            loc[0], loc[1], ball_x_norm, ball_y_norm, is_attacking,
        ))

    # Add ball node
    node_feats.append(sb_to_ussf_ball_features(ball_x_norm, ball_y_norm))
    x = torch.tensor(node_feats, dtype=torch.float32)
    n_nodes = x.shape[0]

    # Dense edges with self-loops
    edge_index = build_ussf_dense_edges(n_nodes)
    n_edges = edge_index.shape[1]

    # Build edge features
    edge_feats = []
    for e in range(n_edges):
        i = edge_index[0, e].item()
        j = edge_index[1, e].item()
        if i == j:
            edge_feats.append(list(_SELF_LOOP_EDGE))
        else:
            edge_feats.append(build_ussf_edge_features_vec(
                x[i, 0].item(), x[i, 1].item(),
                x[j, 0].item(), x[j, 1].item(),
                x[i, 4].item(), x[j, 4].item(),  # vel_mag (both 0)
                x[i, 2].item(), x[i, 3].item(),  # vx_unit, vy_unit (both 0)
                x[j, 2].item(), x[j, 3].item(),
            ))
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    # Receiver mask: attacking outfield players (exclude keepers and ball)
    receiver_mask_list = []
    for player in freeze_frame:
        is_atk = player["teammate"]
        is_gk = player["keeper"]
        receiver_mask_list.append(is_atk and not is_gk)
    receiver_mask_list.append(False)  # ball node
    receiver_mask = torch.tensor(receiver_mask_list, dtype=torch.bool)

    # No receiver labels from StatsBomb freeze-frames
    receiver_label = torch.zeros(n_nodes, dtype=torch.float32)
    has_receiver_label = False

    # Shot / goal labels
    shot_label = 1 if corner_event["lead_to_shot"] else 0
    goal_label = 1 if corner_event["lead_to_goal"] else 0

    corner_side = 1.0 if corner_event["corner_side"] == "right" else 0.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=has_receiver_label,
        shot_label=shot_label,
        goal_label=goal_label,
        corner_side=corner_side,
        match_id=str(corner_event["match_id"]),
        corner_id=corner_event["event_uuid"],
        detection_rate=len(freeze_frame) / 22.0,  # proxy
        source="statsbomb",
    )


def build_dataset(min_players: int = MIN_PLAYERS) -> list:
    """Build full StatsBomb USSF-aligned dataset.

    Scans all 3464 match event files and matches to 323 freeze-frame files.
    Returns list of PyG Data objects.
    """
    print("Loading corner events from all event files...")
    corner_events = load_corner_events()
    print(f"  Found {len(corner_events)} corner kicks with freeze-frames")

    # Build graphs, filtering by player count
    graphs = []
    skipped_few_players = 0

    for evt in corner_events:
        ff = evt["freeze_frame"]
        if len(ff) < min_players:
            skipped_few_players += 1
            continue

        graph = build_statsbomb_ussf_graph(evt, ff)
        graphs.append(graph)

    print(f"\nBuilt {len(graphs)} graphs")
    print(f"  Skipped (<{min_players} players): {skipped_few_players}")

    # Summary
    n_shots = sum(1 for g in graphs if g.shot_label == 1)
    match_ids = sorted(set(str(g.match_id) for g in graphs))
    match_counts = Counter(str(g.match_id) for g in graphs)
    player_counts = [g.x.shape[0] - 1 for g in graphs]  # minus ball node

    print(f"\nDataset summary:")
    print(f"  Corners: {len(graphs)}")
    print(f"  Matches: {len(match_ids)} (LOMO folds)")
    print(f"  Shots: {n_shots}/{len(graphs)} ({100*n_shots/len(graphs):.1f}%)")
    print(f"  Players/frame: min={min(player_counts)}, max={max(player_counts)}, "
          f"mean={np.mean(player_counts):.1f}")
    print(f"  Corners/match: min={min(match_counts.values())}, "
          f"max={max(match_counts.values())}, "
          f"mean={np.mean(list(match_counts.values())):.1f}")

    return graphs


def grouped_kfold_cv(
    dataset: list,
    n_folds: int = 10,
    seed: int = 42,
    device=None,
    verbose: bool = True,
) -> dict:
    """Grouped k-fold CV where groups = match_id (no match leaks across folds).

    Same guarantees as LOMO (no match in both train and test) but with k folds
    instead of N_matches folds, making it feasible for large numbers of matches.
    """
    from torch_geometric.loader import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    pretrained_path = str(PRETRAINED_PATH) if PRETRAINED_PATH.exists() else None

    # Group matches into k folds
    match_ids = sorted(set(str(g.match_id) for g in dataset))
    rng = np.random.RandomState(seed)
    rng.shuffle(match_ids)
    fold_matches = [match_ids[i::n_folds] for i in range(n_folds)]

    fold_results = []

    for fold_idx in range(n_folds):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        test_matches = set(fold_matches[fold_idx])
        val_matches = set(fold_matches[(fold_idx + 1) % n_folds])

        test_data = [g for g in dataset if str(g.match_id) in test_matches]
        val_data = [g for g in dataset if str(g.match_id) in val_matches]
        train_data = [g for g in dataset
                      if str(g.match_id) not in test_matches
                      and str(g.match_id) not in val_matches]

        if not test_data or sum(g.shot_label for g in test_data) == 0:
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_folds}: skipped (no test shots)")
            continue

        if verbose:
            n_train_shots = sum(1 for g in train_data if g.shot_label == 1)
            n_test_shots = sum(1 for g in test_data if g.shot_label == 1)
            print(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_data)} "
                  f"({n_train_shots} shots), val={len(val_data)}, "
                  f"test={len(test_data)} ({n_test_shots} shots)")

        model = build_model(
            backbone_mode="ussf_aligned",
            pretrained_path=pretrained_path,
            freeze=True,
            receiver_hidden=RECEIVER_HIDDEN,
            receiver_dropout=RECEIVER_DROPOUT,
            shot_hidden=SHOT_HIDDEN,
            shot_dropout=SHOT_DROPOUT,
            linear_heads=False,
        ).to(device)

        model, loss_history = train_fold(
            model, train_data, val_data, device,
            receiver_lr=RECEIVER_LR,
            receiver_epochs=RECEIVER_EPOCHS,
            receiver_patience=RECEIVER_PATIENCE,
            receiver_weight_decay=RECEIVER_WEIGHT_DECAY,
            shot_lr=SHOT_LR,
            shot_epochs=SHOT_EPOCHS,
            shot_patience=SHOT_PATIENCE,
            shot_weight_decay=SHOT_WEIGHT_DECAY,
            shot_pos_weight=SHOT_POS_WEIGHT,
            batch_size=BATCH_SIZE,
        )

        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        receiver_metrics = eval_receiver(model, test_loader, device)
        shot_oracle = eval_shot(model, test_loader, device, receiver_mode="oracle")
        shot_predicted = eval_shot(model, test_loader, device, receiver_mode="predicted")
        shot_unconditional = eval_shot(model, test_loader, device, receiver_mode="none")

        fold_result = {
            "fold_idx": fold_idx,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "receiver": receiver_metrics,
            "shot_oracle": shot_oracle,
            "shot_predicted": shot_predicted,
            "shot_unconditional": shot_unconditional,
        }
        fold_results.append(fold_result)

        if verbose:
            auc_o = shot_oracle.get("auc", float("nan"))
            print(f"    Shot AUC (oracle): {auc_o:.3f}")

    # Aggregate
    aggregated = {
        "receiver": compute_receiver_metrics(fold_results),
        "shot_oracle": compute_shot_metrics(fold_results, "oracle"),
        "shot_predicted": compute_shot_metrics(fold_results, "predicted"),
        "shot_unconditional": compute_shot_metrics(fold_results, "unconditional"),
    }

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
        "n_folds": n_folds,
        "n_corners": len(dataset),
        "n_matches": len(match_ids),
        "cv_method": f"grouped_{n_folds}fold",
    }


def permutation_test_kfold(
    dataset: list,
    n_folds: int = 10,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
    device=None,
    verbose: bool = True,
) -> dict:
    """Permutation test using grouped k-fold CV instead of LOMO."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: Shot AUC (grouped {n_folds}-fold, n={n_permutations})")
        print(f"{'=' * 60}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_results = grouped_kfold_cv(dataset, n_folds=n_folds, seed=seed,
                                     device=device, verbose=verbose)
    real_metric = real_results["aggregated"]["shot_oracle"]["auc_mean"]

    if verbose:
        print(f"\nReal AUC (oracle): {real_metric:.4f}")

    # Null distribution
    null_metrics = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_results = grouped_kfold_cv(shuffled, n_folds=n_folds, seed=seed,
                                         device=device, verbose=False)
        null_metric = perm_results["aggregated"]["shot_oracle"]["auc_mean"]
        null_metrics.append(null_metric)

    null_metrics = np.array(null_metrics)
    p_value = (np.sum(null_metrics >= real_metric) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_metrics.mean():.4f}, "
              f"std={null_metrics.std():.4f}")
        print(f"p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return {
        "metric": "shot_auc_oracle",
        "real_metric": float(real_metric),
        "null_distribution": null_metrics.tolist(),
        "null_mean": float(null_metrics.mean()),
        "null_std": float(null_metrics.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": bool(p_value < 0.05),
        "cv_method": f"grouped_{n_folds}fold",
        "n_folds": n_folds,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Option A: USSF pipeline on StatsBomb freeze-frames",
    )
    parser.add_argument("--permutation", action="store_true",
                        help="Run permutation test (N=100)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run LOMO evaluation only (no permutation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-players", type=int, default=MIN_PLAYERS,
                        help="Minimum players in freeze-frame")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of folds for grouped k-fold CV")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("Option A: USSF Pipeline on StatsBomb Freeze-Frames")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Seed: {args.seed}")
    print(f"Min players: {args.min_players}")
    print(f"CV: grouped {args.n_folds}-fold (match-grouped, no leakage)")

    # Build dataset
    graphs = build_dataset(min_players=args.min_players)

    # Save dataset for reproducibility
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_path = RESULTS_DIR / "statsbomb_ussf_graphs.pkl"
    with open(ds_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"\nSaved dataset: {ds_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.eval_only or not args.permutation:
        # Run grouped k-fold evaluation
        print(f"\n{'=' * 60}")
        print(f"Grouped {args.n_folds}-Fold Cross-Validation")
        print(f"{'=' * 60}")
        results = grouped_kfold_cv(graphs, n_folds=args.n_folds, seed=args.seed,
                                    device=device, verbose=True)
        save_results(results, name="statsbomb_ussf_kfold", output_dir=str(RESULTS_DIR))

        agg = results["aggregated"]
        print(f"\n{'=' * 60}")
        print("STATSBOMB USSF RESULTS")
        print(f"{'=' * 60}")
        print(f"Shot AUC (oracle):       {agg['shot_oracle']['auc_mean']:.4f}")
        print(f"Shot AUC (unconditional): {agg['shot_unconditional']['auc_mean']:.4f}")

    if args.permutation:
        # Run permutation test with grouped k-fold
        perm_result = permutation_test_kfold(
            graphs, n_folds=args.n_folds,
            n_permutations=args.n_permutations,
            seed=args.seed, device=device, verbose=True,
        )

        perm_result["dataset"] = "statsbomb"
        perm_result["n_corners"] = len(graphs)
        perm_result["n_matches"] = len(set(str(g.match_id) for g in graphs))
        perm_result["timestamp"] = str(datetime.now())

        json_path = RESULTS_DIR / "statsbomb_ussf_perm_shot.json"
        with open(json_path, "w") as f:
            json.dump(perm_result, f, indent=2)
        print(f"\nSaved: {json_path}")

        pkl_path = RESULTS_DIR / "statsbomb_ussf_perm_shot.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(perm_result, f)
        print(f"Saved: {pkl_path}")

        print(f"\n{'=' * 60}")
        print("PERMUTATION TEST RESULT")
        print(f"{'=' * 60}")
        print(f"Real AUC: {perm_result['real_metric']:.4f}")
        print(f"Null: mean={perm_result['null_mean']:.4f}, std={perm_result['null_std']:.4f}")
        print(f"p-value: {perm_result['p_value']:.4f} "
              f"{'***' if perm_result['p_value'] < 0.01 else '**' if perm_result['p_value'] < 0.05 else '(not significant)'}")


if __name__ == "__main__":
    main()
