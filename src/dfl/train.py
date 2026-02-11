"""Training utilities for corner kick ST-GNN.

This module provides:
- Open-play sequence extraction for pretraining
- Spatial GNN pretraining on shot prediction
- Fine-tuning on corner kick dataset
- Leave-one-match-out cross-validation
- Multi-task loss computation
- Velocity ablation experiment utilities
"""

import copy
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.dfl.model import CornerKickPredictor
from src.dfl.graph_construction import frame_to_graph
from src.dfl.data_loading import compute_velocities


def extract_open_play_sequences(
    tracking_dataset,
    event_dataset,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    lookahead_seconds: float = 6.0,
    fps: int = 25,
) -> List[Dict[str, Any]]:
    """Extract overlapping windows from open play for pretraining.

    Labels each window with: did a shot happen within lookahead_seconds?

    Args:
        tracking_dataset: kloppy TrackingDataset
        event_dataset: kloppy EventDataset
        window_seconds: Duration of each window
        stride_seconds: Stride between windows
        lookahead_seconds: How far ahead to look for shots
        fps: Frame rate of tracking data

    Returns:
        List of dicts with keys:
            - 'frames': List of tracking frames
            - 'shot_label': 1 if shot within lookahead, else 0
            - 'start_time': Timestamp of window start
    """
    sequences = []
    window_frames = int(window_seconds * fps)
    stride_frames = int(stride_seconds * fps)

    all_frames = list(tracking_dataset.records)

    # Find all shot events
    shot_times = []
    for event in event_dataset.events:
        event_type_str = str(event.event_type).lower()
        if "shot" in event_type_str:
            if hasattr(event.timestamp, "total_seconds"):
                shot_times.append(event.timestamp.total_seconds())
            else:
                shot_times.append(float(event.timestamp))

    for start_idx in range(0, len(all_frames) - window_frames, stride_frames):
        window = all_frames[start_idx : start_idx + window_frames]

        if not window:
            continue

        # Get window end time
        end_frame = window[-1]
        if hasattr(end_frame.timestamp, "total_seconds"):
            window_end_time = end_frame.timestamp.total_seconds()
        else:
            window_end_time = float(end_frame.timestamp)

        # Label: will there be a shot within lookahead_seconds?
        shot_within_lookahead = any(
            0 < (st - window_end_time) < lookahead_seconds for st in shot_times
        )

        sequences.append(
            {
                "frames": window,
                "shot_label": 1 if shot_within_lookahead else 0,
                "start_time": window[0].timestamp,
            }
        )

    return sequences


def pretrain_spatial_gnn(
    model: CornerKickPredictor,
    open_play_sequences: List[Dict[str, Any]],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 3.0,
) -> CornerKickPredictor:
    """Pretrain only the spatial GNN on open-play data.

    Uses single-frame prediction with a temporary pretraining head.
    The pretrain head projects from gnn_out (32) to 1 for shot prediction.

    Args:
        model: CornerKickPredictor model
        open_play_sequences: Output of extract_open_play_sequences or
            list of dicts with 'graph', 'edge_index', 'shot_label'
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        pos_weight: Weight for positive class (handle imbalance)

    Returns:
        Trained model with pretrained spatial_gnn weights
    """
    # Create a temporary pretraining head matching spatial_gnn output dimension
    gnn_out_dim = model.spatial_gnn.out_channels
    pretrain_head = nn.Linear(gnn_out_dim, 1)

    optimizer = torch.optim.Adam(
        list(model.spatial_gnn.parameters()) + list(pretrain_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight])
    )

    model.train()
    pretrain_head.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for seq in open_play_sequences:
            # Handle both formats: full sequences with frames or simple dicts
            if "graph" in seq:
                # Simple format for testing
                x = seq["graph"]
                edge_index = seq["edge_index"]
            elif "frames" in seq:
                # Full format with frames - use middle frame
                frames = seq["frames"]
                mid_idx = len(frames) // 2
                velocities = compute_velocities(frames, fps=25)

                # Create a dummy corner event for graph construction
                class DummyEvent:
                    team = None

                graph = frame_to_graph(
                    frames[mid_idx], velocities[mid_idx], DummyEvent()
                )
                x = graph.x
                edge_index = graph.edge_index
            else:
                continue

            # Forward through spatial GNN and pretrain head
            emb = model.spatial_gnn(x, edge_index)
            pred = pretrain_head(emb)

            target = torch.tensor([[float(seq["shot_label"])]])
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model


def leave_one_match_out_split(
    dataset: List[Dict[str, Any]]
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Split dataset using leave-one-match-out cross-validation.

    Args:
        dataset: List of samples with 'match_id' key

    Returns:
        List of (train_data, test_data) tuples, one per unique match
    """
    matches = sorted(set(d["match_id"] for d in dataset))
    folds = []

    for test_match in matches:
        train_data = [d for d in dataset if d["match_id"] != test_match]
        test_data = [d for d in dataset if d["match_id"] == test_match]

        if test_data:  # Only include if test set is non-empty
            folds.append((train_data, test_data))

    return folds


def compute_multi_task_loss(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    shot_weight: float = 1.0,
    goal_weight: float = 1.0,
    contact_weight: float = 0.5,
    outcome_weight: float = 0.5,
) -> torch.Tensor:
    """Compute combined multi-task loss.

    Args:
        predictions: Dict with 'shot', 'goal', 'contact', 'outcome' tensors
        labels: Dict with 'shot_binary', 'goal_binary', 'first_contact_team',
            'outcome_class' tensors
        shot_weight: Weight for shot loss
        goal_weight: Weight for goal loss
        contact_weight: Weight for contact loss
        outcome_weight: Weight for outcome loss

    Returns:
        Scalar loss tensor
    """
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Shot loss (already sigmoid in model output)
    shot_target = labels["shot_binary"].float().unsqueeze(-1)
    shot_loss = bce_loss(predictions["shot"], shot_target)

    # Goal loss
    goal_target = labels["goal_binary"].float().unsqueeze(-1)
    goal_loss = bce_loss(predictions["goal"], goal_target)

    # Contact loss (2-class cross entropy)
    contact_loss = ce_loss(predictions["contact"], labels["first_contact_team"])

    # Outcome loss (N-class cross entropy)
    outcome_loss = ce_loss(predictions["outcome"], labels["outcome_class"])

    total_loss = (
        shot_weight * shot_loss
        + goal_weight * goal_loss
        + contact_weight * contact_loss
        + outcome_weight * outcome_loss
    )

    return total_loss


def finetune_on_corners(
    model: CornerKickPredictor,
    corner_dataset: List[Dict[str, Any]],
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Fine-tune the full model on corner kicks with cross-validation.

    Uses leave-one-match-out cross-validation.

    Args:
        model: Pretrained CornerKickPredictor model
        corner_dataset: List of corner samples from build_corner_dataset
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        List of dicts with fold results including 'test_match' and metrics
    """
    folds = leave_one_match_out_split(corner_dataset)
    results = []

    # Map string labels to indices
    contact_map = {"attacking": 0, "defending": 1, "unknown": 0}
    outcome_map = {
        "goal": 0,
        "shot_saved": 1,
        "shot_blocked": 2,
        "clearance": 3,
        "ball_receipt": 4,
        "other": 5,
    }

    for train_data, test_data in folds:
        if not train_data or not test_data:
            continue

        test_match = test_data[0]["match_id"]

        # Clone model for this fold
        fold_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            fold_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Training loop
        fold_model.train()
        for epoch in range(epochs):
            for sample in train_data:
                graphs = sample["graphs"]
                labels = sample["labels"]

                # Prepare labels as tensors
                label_tensors = {
                    "shot_binary": torch.tensor([labels["shot_binary"]]),
                    "goal_binary": torch.tensor([labels["goal_binary"]]),
                    "first_contact_team": torch.tensor(
                        [contact_map.get(labels["first_contact_team"], 0)]
                    ),
                    "outcome_class": torch.tensor(
                        [outcome_map.get(labels["outcome_class"], 5)]
                    ),
                }

                # Forward pass
                predictions = fold_model([graphs])

                loss = compute_multi_task_loss(predictions, label_tensors)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on test fold
        fold_model.eval()
        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for sample in test_data:
                graphs = sample["graphs"]
                predictions = fold_model([graphs])

                test_predictions.append(predictions["shot"].item())
                test_labels.append(sample["labels"]["shot_binary"])

        results.append(
            {
                "test_match": test_match,
                "predictions": test_predictions,
                "labels": test_labels,
            }
        )

    return results


def zero_out_velocity_features(
    dataset: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Create a copy of dataset with velocity features (vx, vy) zeroed out.

    Velocity features are at indices 2 and 3 in the 8-feature node vector:
    [x, y, vx, vy, team_flag, is_kicker, dist_to_goal, dist_to_ball]

    Args:
        dataset: List of corner samples

    Returns:
        New dataset with vx=0, vy=0 for all nodes
    """
    modified_dataset = []

    for sample in dataset:
        new_sample = {
            "labels": sample["labels"],
            "match_id": sample["match_id"],
            "corner_time": sample.get("corner_time", 0.0),
            "graphs": [],
        }

        for graph in sample["graphs"]:
            # Clone the node features
            new_x = graph.x.clone()
            # Zero out vx (index 2) and vy (index 3)
            new_x[:, 2] = 0.0
            new_x[:, 3] = 0.0

            new_graph = Data(
                x=new_x,
                edge_index=graph.edge_index.clone(),
                pos=graph.pos.clone() if hasattr(graph, "pos") and graph.pos is not None else None,
            )

            # Copy other attributes
            if hasattr(graph, "frame_idx"):
                new_graph.frame_idx = graph.frame_idx
            if hasattr(graph, "relative_time"):
                new_graph.relative_time = graph.relative_time

            new_sample["graphs"].append(new_graph)

        modified_dataset.append(new_sample)

    return modified_dataset


def run_ablation(
    corner_dataset: List[Dict[str, Any]],
    epochs: int = 100,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    """Run velocity ablation experiment.

    Compares:
    - Condition A: Position-only features (vx=0, vy=0)
    - Condition B: Position+Velocity features (full 8 features)

    Args:
        corner_dataset: List of corner samples
        epochs: Training epochs per fold
        lr: Learning rate

    Returns:
        Dict with:
            - 'position_only': Results from Condition A
            - 'position_velocity': Results from Condition B
            - 'delta': Difference in mean metrics
    """
    # Condition A: Position-only (zero out velocities)
    dataset_no_vel = zero_out_velocity_features(corner_dataset)
    model_no_vel = CornerKickPredictor(node_features=8)
    results_no_vel = finetune_on_corners(
        model_no_vel, dataset_no_vel, epochs=epochs, lr=lr
    )

    # Condition B: Full features
    dataset_full = corner_dataset
    model_full = CornerKickPredictor(node_features=8)
    results_full = finetune_on_corners(
        model_full, dataset_full, epochs=epochs, lr=lr
    )

    return {
        "position_only": results_no_vel,
        "position_velocity": results_full,
        "delta": None,  # Could compute AUC difference here
    }
