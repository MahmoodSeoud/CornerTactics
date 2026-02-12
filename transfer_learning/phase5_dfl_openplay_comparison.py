#!/usr/bin/env python3
"""
Phase 5: DFL Open-Play Pretraining Comparison
==============================================

Compare USSF counterattack pretraining against DFL open-play pretraining
for corner kick shot prediction.

Key Question: Is USSF counterattack pretraining better or worse than
pretraining on open-play sequences from the same DFL data source?

Expected Result: DFL open-play pretraining should win because:
1. Same coordinate system (no distribution shift)
2. Same tracking precision
3. Includes dense situations (set pieces, congested play)

Pretraining Task: Shot occurrence prediction
- At each open-play frame, predict if a shot occurs within next 5 seconds
- This is similar to USSF counterattack success prediction

Experimental Conditions:
    G: DFL pretrained + dense adjacency + frozen backbone
    H: DFL pretrained + dense adjacency + unfrozen (fine-tuned, lr=1e-5)

Compare against Phase 3 conditions:
    A: USSF pretrained + dense + frozen (best USSF: 0.57 AUC)
    F: Majority baseline (0.50 AUC)
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')
warnings.filterwarnings('ignore', message='.*torch-sparse.*')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dfl.data_loading import load_tracking_data, load_event_data, compute_velocities

# Configuration
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"
RESULTS_DIR = Path(__file__).parent / "results"
DFL_DATA_DIR = Path("/home/mseo/CornerTactics/data/dfl")

# DFL matches
DFL_MATCHES = [
    "DFL-MAT-J03WMX",
    "DFL-MAT-J03WN1",
    "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY",
    "DFL-MAT-J03WQQ",
    "DFL-MAT-J03WR9",
]

# Pitch dimensions
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Pretraining parameters
SHOT_HORIZON_SECONDS = 5.0  # Predict if shot within this window
SAMPLE_INTERVAL_SECONDS = 1.0  # Sample one frame per second
MIN_PLAYERS_PER_FRAME = 20  # Require at least this many visible players


class CounterattackGNN(nn.Module):
    """
    CrystalConv GNN for pretraining (same architecture as Phase 1).
    """

    def __init__(
        self,
        node_features: int = 12,
        edge_features: int = 6,
        hidden_channels: int = 128,
        num_conv_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels

        self.conv1 = CGConv(
            channels=node_features,
            dim=edge_features,
            aggr='add'
        )
        self.lin_in = nn.Linear(node_features, hidden_channels)

        self.convs = nn.ModuleList([
            CGConv(
                channels=hidden_channels,
                dim=edge_features,
                aggr='add'
            )
            for _ in range(num_conv_layers - 1)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.head(x)
        return x

    def get_backbone_state(self) -> Dict:
        return {
            'conv1': self.conv1.state_dict(),
            'lin_in': self.lin_in.state_dict(),
            'convs': [conv.state_dict() for conv in self.convs],
            'config': {
                'node_features': self.node_features,
                'edge_features': self.edge_features,
                'hidden_channels': self.hidden_channels,
                'num_conv_layers': len(self.convs) + 1
            }
        }


class TransferGNN(nn.Module):
    """Transfer learning GNN (same as Phase 3)."""

    def __init__(
        self,
        node_features: int = 12,
        edge_features: int = 6,
        hidden_channels: int = 128,
        num_conv_layers: int = 3,
        head_hidden: int = 32,
        head_dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.freeze_backbone = freeze_backbone

        self.conv1 = CGConv(
            channels=node_features,
            dim=edge_features,
            aggr='add'
        )
        self.lin_in = nn.Linear(node_features, hidden_channels)

        self.convs = nn.ModuleList([
            CGConv(
                channels=hidden_channels,
                dim=edge_features,
                aggr='add'
            )
            for _ in range(num_conv_layers - 1)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def load_pretrained_backbone(self, backbone_state: Dict):
        self.conv1.load_state_dict(backbone_state['conv1'])
        self.lin_in.load_state_dict(backbone_state['lin_in'])
        for i, conv_state in enumerate(backbone_state['convs']):
            self.convs[i].load_state_dict(conv_state)
        print(f"  Loaded pretrained backbone weights")

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.head(x)
        return x

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def normalize_angle(angle_rad: float) -> float:
    """Normalize angle from [-pi, pi] to [0, 1]."""
    return (angle_rad + np.pi) / (2 * np.pi)


def compute_unit_vector(vx: float, vy: float) -> Tuple[float, float, float]:
    """Convert velocity to unit vector and magnitude."""
    magnitude = np.sqrt(vx**2 + vy**2)
    if magnitude < 1e-8:
        return 1.0, 0.0, 0.0
    return vx / magnitude, vy / magnitude, magnitude


def frame_to_ussf_graph(
    frame,
    velocities: Dict[Any, Dict[str, float]],
    home_team_attacking: bool,
    adj_type: str = 'dense'
) -> Optional[Data]:
    """
    Convert a single tracking frame to USSF-compatible graph.

    Args:
        frame: kloppy Frame object
        velocities: Dict mapping player_id -> {"vx": float, "vy": float}
        home_team_attacking: Whether home team is attacking towards positive x
        adj_type: 'dense' for fully connected

    Returns:
        PyG Data object or None if invalid frame
    """
    # Collect all player data
    players = []
    ball_pos = None

    # Get ball position (kloppy uses normalized coordinates [0,1])
    if frame.ball_coordinates:
        ball_x = frame.ball_coordinates.x
        ball_y = frame.ball_coordinates.y
        if ball_x is not None and ball_y is not None:
            # kloppy already returns normalized coordinates
            ball_pos = (ball_x, ball_y)

    if ball_pos is None:
        return None

    # Process players
    # In kloppy, player_id is a Player object with .team attribute
    for player, player_data in frame.players_data.items():
        if player_data.coordinates is None:
            continue

        x = player_data.coordinates.x
        y = player_data.coordinates.y

        if x is None or y is None:
            continue

        # kloppy returns normalized coordinates [0,1]
        x_norm = np.clip(x, 0.0, 1.0)
        y_norm = np.clip(y, 0.0, 1.0)

        # Get velocity (velocities dict is keyed by player object)
        vel = velocities.get(player, {"vx": 0.0, "vy": 0.0})
        vx = vel["vx"]  # Already in m/s from compute_velocities
        vy = vel["vy"]

        # Determine if attacking - player is a Player object with .team.ground
        try:
            is_home = player.team.ground.value == "home"
        except (AttributeError, TypeError):
            is_home = True  # Default assumption
        is_attacking = 1.0 if (is_home == home_team_attacking) else 0.0

        players.append({
            'x_norm': x_norm,
            'y_norm': y_norm,
            'vx': vx,
            'vy': vy,
            'is_attacking': is_attacking,
            'is_ball': False
        })

    if len(players) < MIN_PLAYERS_PER_FRAME:
        return None

    # Add ball as a node
    players.append({
        'x_norm': ball_pos[0],
        'y_norm': ball_pos[1],
        'vx': 0.0,  # Ball velocity not tracked in simple version
        'vy': 0.0,
        'is_attacking': 0.0,  # Ball is neutral
        'is_ball': True
    })

    # Build node features (USSF format: 12 features)
    goal_x = 1.0  # Goal at positive x end
    goal_y = 0.5  # Center of goal
    max_velocity = 10.0  # m/s

    node_features = []
    for p in players:
        x_final = p['x_norm']
        y_final = p['y_norm']
        vx = p['vx']
        vy = p['vy']

        # Velocity unit vector and magnitude
        vx_unit, vy_unit, vel_mag = compute_unit_vector(vx, vy)
        vel_mag_norm = np.clip(vel_mag / max_velocity, 0.0, 1.0)
        vel_angle = np.arctan2(vy_unit, vx_unit)
        vel_angle_norm = normalize_angle(vel_angle)

        # Distance and angle to goal
        dx_goal = goal_x - x_final
        dy_goal = goal_y - y_final
        dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        dist_goal_norm = np.clip(dist_goal / np.sqrt(2), 0.0, 1.0)
        angle_goal = np.arctan2(dy_goal, dx_goal)
        angle_goal_norm = normalize_angle(angle_goal)

        # Distance and angle to ball
        dx_ball = ball_pos[0] - x_final
        dy_ball = ball_pos[1] - y_final
        dist_ball = np.sqrt(dx_ball**2 + dy_ball**2)
        dist_ball_norm = np.clip(dist_ball / np.sqrt(2), 0.0, 1.0)
        angle_ball = np.arctan2(dy_ball, dx_ball)
        angle_ball_norm = normalize_angle(angle_ball)

        node_features.append([
            x_final,           # 0. x
            y_final,           # 1. y
            vx_unit,           # 2. vx (unit vector)
            vy_unit,           # 3. vy (unit vector)
            vel_mag_norm,      # 4. velocity_mag
            vel_angle_norm,    # 5. velocity_angle
            dist_goal_norm,    # 6. dist_goal
            angle_goal_norm,   # 7. angle_goal
            dist_ball_norm,    # 8. dist_ball
            angle_ball_norm,   # 9. angle_ball
            p['is_attacking'], # 10. attacking_team_flag
            0.0,               # 11. potential_receiver (always 0)
        ])

    node_features = np.array(node_features, dtype=np.float32)
    n_nodes = len(node_features)

    # Build adjacency (dense = fully connected)
    if adj_type == 'dense':
        src, dst = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        raise NotImplementedError(f"Adjacency type {adj_type} not implemented")

    # Compute edge features
    edge_attr = []
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e].item(), edge_index[1, e].item()
        ni, nj = node_features[i], node_features[j]

        # Player distance
        dx = nj[0] - ni[0]
        dy = nj[1] - ni[1]
        player_distance = np.sqrt(dx**2 + dy**2)
        player_distance_norm = np.clip(player_distance / np.sqrt(2), 0.0, 1.0)

        # Speed difference
        speed_diff = nj[4] - ni[4]  # Already normalized

        # Positional angle
        pos_angle = np.arctan2(dy, dx)
        pos_sin = (np.sin(pos_angle) + 1) / 2
        pos_cos = (np.cos(pos_angle) + 1) / 2

        # Velocity angle between vectors
        vxi, vyi = ni[2], ni[3]
        vxj, vyj = nj[2], nj[3]
        dot = vxi * vxj + vyi * vyj
        cross = vxi * vyj - vyi * vxj
        vel_angle = np.arctan2(cross, dot)
        vel_sin = (np.sin(vel_angle) + 1) / 2
        vel_cos = (np.cos(vel_angle) + 1) / 2

        edge_attr.append([
            player_distance_norm,
            speed_diff,
            pos_sin,
            pos_cos,
            vel_sin,
            vel_cos,
        ])

    edge_attr = np.array(edge_attr, dtype=np.float32)

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )


def find_shot_times(event_dataset) -> List[Tuple[int, float]]:
    """
    Find all shot events and their timestamps.

    Returns:
        List of (period, timestamp_seconds) tuples
    """
    shot_times = []

    for event in event_dataset.events:
        event_type = str(event.event_type).lower()

        if "shot" in event_type:
            period = event.period.id if hasattr(event.period, 'id') else 1

            # Get timestamp as seconds
            ts = event.timestamp
            if hasattr(ts, 'total_seconds'):
                ts_seconds = ts.total_seconds()
            else:
                ts_seconds = float(ts)

            shot_times.append((period, ts_seconds))

    return shot_times


def extract_openplay_samples(
    match_id: str,
    sample_interval: float = SAMPLE_INTERVAL_SECONDS,
    shot_horizon: float = SHOT_HORIZON_SECONDS,
    adj_type: str = 'dense'
) -> Tuple[List[Data], List[int]]:
    """
    Extract open-play samples from a single match.

    Labels: 1 if shot occurs within shot_horizon seconds, 0 otherwise.

    Args:
        match_id: DFL match identifier
        sample_interval: Sample one frame every N seconds
        shot_horizon: Look-ahead window for shot prediction
        adj_type: Adjacency type for graphs

    Returns:
        Tuple of (graphs, labels)
    """
    print(f"\nProcessing {match_id}...")

    # Load tracking data
    try:
        tracking = load_tracking_data("dfl", DFL_DATA_DIR, match_id)
    except Exception as e:
        print(f"  ERROR loading tracking data: {e}")
        return [], []

    # Load events
    try:
        events = load_event_data("dfl", DFL_DATA_DIR, match_id)
    except Exception as e:
        print(f"  ERROR loading event data: {e}")
        return [], []

    # Find shot times
    shot_times = find_shot_times(events)
    print(f"  Found {len(shot_times)} shots")

    # Convert frames to list for velocity computation
    frames = list(tracking.records)
    print(f"  Total frames: {len(frames)}")

    if len(frames) < 10:
        print(f"  Too few frames, skipping")
        return [], []

    # Compute velocities
    print(f"  Computing velocities...")
    velocities = compute_velocities(frames, fps=25)

    # Determine attacking direction
    # Assume home team attacks towards positive x in first half
    # This is a simplification - in reality we'd check period
    home_team_attacking = True

    # Sample frames at regular intervals
    fps = 25
    frame_interval = int(sample_interval * fps)

    graphs = []
    labels = []

    last_period = None
    period_start_idx = 0

    print(f"  Extracting samples...")
    for idx in range(0, len(frames), frame_interval):
        frame = frames[idx]

        # Skip if no timestamp
        if not hasattr(frame, 'timestamp'):
            continue

        ts = frame.timestamp
        if hasattr(ts, 'total_seconds'):
            ts_seconds = ts.total_seconds()
        else:
            ts_seconds = float(ts) if ts is not None else 0.0

        period = frame.period.id if hasattr(frame, 'period') and hasattr(frame.period, 'id') else 1

        # Flip attacking direction for second half
        attacking_dir = home_team_attacking if period == 1 else not home_team_attacking

        # Create graph
        graph = frame_to_ussf_graph(
            frame,
            velocities[idx],
            home_team_attacking=attacking_dir,
            adj_type=adj_type
        )

        if graph is None:
            continue

        # Determine label: is there a shot within next shot_horizon seconds?
        label = 0
        for shot_period, shot_ts in shot_times:
            if shot_period == period:
                time_to_shot = shot_ts - ts_seconds
                if 0 < time_to_shot <= shot_horizon:
                    label = 1
                    break

        graphs.append(graph)
        labels.append(label)

    print(f"  Extracted {len(graphs)} samples ({sum(labels)} positive)")
    return graphs, labels


def extract_all_openplay_data(
    matches: List[str],
    adj_type: str = 'dense'
) -> Tuple[List[Data], List[int], List[str]]:
    """
    Extract open-play samples from all matches.

    Returns:
        Tuple of (graphs, labels, match_ids)
    """
    all_graphs = []
    all_labels = []
    all_match_ids = []

    for match_id in matches:
        graphs, labels = extract_openplay_samples(match_id, adj_type=adj_type)
        all_graphs.extend(graphs)
        all_labels.extend(labels)
        all_match_ids.extend([match_id] * len(graphs))

    return all_graphs, all_labels, all_match_ids


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None
) -> float:
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(), batch.y.squeeze(),
                pos_weight=pos_weight.to(device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples if n_samples > 0 else 0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs])

            preds = (probs > 0.5).astype(int)

            labels = batch.y.squeeze().cpu().numpy()
            if labels.ndim == 0:
                labels = np.array([labels])

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return {
        'loss': total_loss / n_samples if n_samples > 0 else 0,
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': n_samples,
        'n_positive': int(all_labels.sum()),
    }


def train_dfl_backbone(
    graphs: List[Data],
    labels: List[int],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
    adj_type: str = 'dense'
) -> Tuple[CounterattackGNN, Dict]:
    """
    Train CrystalConv backbone on DFL open-play data.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Add labels to graphs
    for g, y in zip(graphs, labels):
        g.y = torch.tensor([y], dtype=torch.float32)

    # Split data
    indices = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=15, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42,
        stratify=[labels[i] for i in train_idx]
    )

    train_data = [graphs[i] for i in train_idx]
    val_data = [graphs[i] for i in val_idx]
    test_data = [graphs[i] for i in test_idx]

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"Positive rate: train={sum(labels[i] for i in train_idx)/len(train_idx):.3f}, "
          f"val={sum(labels[i] for i in val_idx)/len(val_idx):.3f}, "
          f"test={sum(labels[i] for i in test_idx)/len(test_idx):.3f}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Compute class weights
    n_pos = sum(labels[i] for i in train_idx)
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])

    # Create model
    model = CounterattackGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=128,
        num_conv_layers=3,
        dropout=0.5
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_val_auc': 0
    }

    best_val_auc = 0
    best_model_state = None

    print(f"\nTraining DFL backbone for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics['auc'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            history['best_val_auc'] = best_val_auc
            marker = '*'
        else:
            marker = ''

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f} {marker}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    test_metrics = evaluate(model, test_loader, device)
    history['test_metrics'] = test_metrics

    print("-" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {history['best_epoch']}")
    print(f"Test metrics:")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")

    return model, history


def run_transfer_experiment(
    condition: str,
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    backbone_path: Path,
    freeze_backbone: bool = True,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 8,
    patience: int = 10,
    device: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """Run a single transfer experiment condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Condition {condition}: DFL pretrained, frozen={freeze_backbone}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Class weights
    train_labels = [d.y.item() for d in train_data]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])

    print(f"Training samples: {len(train_data)} ({n_pos} positive)")

    # Create model
    model = TransferGNN(
        node_features=12,
        edge_features=6,
        hidden_channels=128,
        num_conv_layers=3,
        head_hidden=32,
        head_dropout=0.3,
        freeze_backbone=freeze_backbone
    ).to(device)

    # Load pretrained weights
    if backbone_path.exists():
        backbone_state = torch.load(backbone_path, map_location=device)
        model.load_pretrained_backbone(backbone_state)
    else:
        print(f"  WARNING: Weights not found at {backbone_path}")

    print(f"Trainable params: {model.count_trainable_params():,}")

    # Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'best_epoch': 0,
        'best_val_auc': 0.0
    }

    best_val_auc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch
            history['best_val_auc'] = best_val_auc
            epochs_without_improvement = 0
            marker = '*'
        else:
            epochs_without_improvement += 1
            marker = ''

        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:2d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.3f} {marker}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device)

    print(f"\nResults for Condition {condition}:")
    print(f"  Best val AUC: {history['best_val_auc']:.4f}")
    print(f"  Test AUC:     {test_metrics['auc']:.4f}")
    print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")

    return {
        'condition': condition,
        'config': {
            'pretrained': 'dfl',
            'freeze_backbone': freeze_backbone,
            'lr': lr,
        },
        'history': history,
        'test_metrics': test_metrics
    }


def prepare_corner_data(adj_type: str = 'dense') -> List[Data]:
    """Load and prepare corner kick data for fine-tuning."""
    data_path = DATA_DIR / f"dfl_corners_ussf_format_{adj_type}.pkl"

    with open(data_path, 'rb') as f:
        corners = pickle.load(f)

    data_list = []
    for sample in corners:
        graph = sample['graphs'][0]
        label = float(sample['labels']['shot_binary'])

        pyg_data = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=torch.tensor([label], dtype=torch.float32)
        )
        pyg_data.match_id = sample['match_id']
        data_list.append(pyg_data)

    return data_list


def match_based_split(
    data_list: List[Data],
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Split data keeping matches together."""
    np.random.seed(seed)

    # Group by match
    match_to_indices = defaultdict(list)
    for i, d in enumerate(data_list):
        match_to_indices[d.match_id].append(i)

    # Stats per match
    match_stats = {}
    for match_id, indices in match_to_indices.items():
        labels = [data_list[i].y.item() for i in indices]
        match_stats[match_id] = {
            'indices': indices,
            'count': len(indices),
            'positives': sum(labels),
        }

    # Select val/test matches
    match_ids = list(match_to_indices.keys())
    np.random.shuffle(match_ids)

    matches_with_pos = [m for m in match_ids if match_stats[m]['positives'] > 0]

    if len(matches_with_pos) >= 2:
        val_match = matches_with_pos[0]
        test_match = matches_with_pos[1]
    else:
        val_match = match_ids[0]
        test_match = match_ids[1]

    train_matches = [m for m in match_ids if m not in [val_match, test_match]]

    train_idx = []
    for m in train_matches:
        train_idx.extend(match_to_indices[m])
    val_idx = match_to_indices[val_match]
    test_idx = match_to_indices[test_match]

    return train_idx, val_idx, test_idx


def run_multi_seed_comparison(
    corners: List[Data],
    dfl_backbone_path: Path,
    ussf_backbone_path: Path,
    seeds: List[int],
    epochs: int = 50,
    device: Optional[str] = None
) -> Dict:
    """
    Run comparison across multiple seeds.

    Conditions:
        G: DFL pretrained + frozen
        H: DFL pretrained + unfrozen
        A: USSF pretrained + frozen (from Phase 3)
    """
    all_seed_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        # Split
        train_idx, val_idx, test_idx = match_based_split(corners, seed=seed)

        train_data = [corners[i] for i in train_idx]
        val_data = [corners[i] for i in val_idx]
        test_data = [corners[i] for i in test_idx]

        print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # G: DFL pretrained + frozen
        result_g = run_transfer_experiment(
            condition='G',
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            backbone_path=dfl_backbone_path,
            freeze_backbone=True,
            lr=1e-4,
            epochs=epochs,
            device=device,
            seed=seed
        )
        all_seed_results[seed]['G'] = result_g

        # H: DFL pretrained + unfrozen
        result_h = run_transfer_experiment(
            condition='H',
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            backbone_path=dfl_backbone_path,
            freeze_backbone=False,
            lr=1e-5,
            epochs=epochs,
            device=device,
            seed=seed
        )
        all_seed_results[seed]['H'] = result_h

        # A: USSF pretrained + frozen (for comparison)
        result_a = run_transfer_experiment(
            condition='A',
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            backbone_path=ussf_backbone_path,
            freeze_backbone=True,
            lr=1e-4,
            epochs=epochs,
            device=device,
            seed=seed
        )
        all_seed_results[seed]['A'] = result_a

    # Aggregate
    aggregated = {}
    for condition in ['G', 'H', 'A']:
        aucs = [all_seed_results[seed][condition]['test_metrics']['auc'] for seed in seeds]
        accs = [all_seed_results[seed][condition]['test_metrics']['accuracy'] for seed in seeds]

        aggregated[condition] = {
            'test_auc_mean': np.mean(aucs),
            'test_auc_std': np.std(aucs),
            'test_acc_mean': np.mean(accs),
            'test_acc_std': np.std(accs),
            'aucs': aucs,
        }

    return {
        'per_seed': all_seed_results,
        'aggregated': aggregated
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 5: DFL Open-Play Comparison')
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract open-play data, do not train')
    parser.add_argument('--train-backbone-only', action='store_true',
                        help='Only train DFL backbone, do not run transfer experiments')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip extraction, use cached data')
    parser.add_argument('--skip-backbone', action='store_true',
                        help='Skip backbone training, use cached weights')
    parser.add_argument('--epochs-pretrain', type=int, default=100,
                        help='Epochs for pretraining')
    parser.add_argument('--epochs-finetune', type=int, default=50,
                        help='Epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for pretraining')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1234],
                        help='Seeds for multi-seed evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5: DFL Open-Play Pretraining Comparison")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    openplay_cache = DATA_DIR / "dfl_openplay_graphs.pkl"
    dfl_backbone_path = WEIGHTS_DIR / "dfl_backbone_dense.pt"
    ussf_backbone_path = WEIGHTS_DIR / "ussf_backbone_dense.pt"

    # Step 1: Extract open-play data
    if not args.skip_extraction:
        print("\n" + "=" * 60)
        print("Step 1: Extract DFL Open-Play Data")
        print("=" * 60)

        graphs, labels, match_ids = extract_all_openplay_data(DFL_MATCHES)

        print(f"\nTotal extracted: {len(graphs)} graphs, {sum(labels)} positive ({100*sum(labels)/len(labels):.1f}%)")

        # Save
        with open(openplay_cache, 'wb') as f:
            pickle.dump({'graphs': graphs, 'labels': labels, 'match_ids': match_ids}, f)
        print(f"Saved to {openplay_cache}")

        if args.extract_only:
            print("\n--extract-only specified, stopping here")
            return
    else:
        print("\nLoading cached open-play data...")
        with open(openplay_cache, 'rb') as f:
            cached = pickle.load(f)
        graphs = cached['graphs']
        labels = cached['labels']
        print(f"Loaded {len(graphs)} graphs")

    # Step 2: Train DFL backbone
    if not args.skip_backbone:
        print("\n" + "=" * 60)
        print("Step 2: Train DFL Open-Play Backbone")
        print("=" * 60)

        model, history = train_dfl_backbone(
            graphs, labels,
            epochs=args.epochs_pretrain,
            batch_size=args.batch_size,
            lr=1e-3,
            device=args.device
        )

        # Save backbone
        torch.save(model.get_backbone_state(), dfl_backbone_path)
        print(f"Saved backbone to {dfl_backbone_path}")

        # Save history
        history_path = WEIGHTS_DIR / "dfl_openplay_training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        if args.train_backbone_only:
            print("\n--train-backbone-only specified, stopping here")
            return
    else:
        print(f"\nUsing cached backbone from {dfl_backbone_path}")

    # Step 3: Run transfer experiments
    print("\n" + "=" * 60)
    print("Step 3: Transfer Learning Comparison")
    print("=" * 60)

    # Load corner data
    corners = prepare_corner_data(adj_type='dense')
    print(f"Loaded {len(corners)} corners for fine-tuning")

    # Run multi-seed comparison
    results = run_multi_seed_comparison(
        corners=corners,
        dfl_backbone_path=dfl_backbone_path,
        ussf_backbone_path=ussf_backbone_path,
        seeds=args.seeds,
        epochs=args.epochs_finetune,
        device=args.device
    )

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n{'Condition':<12} {'Description':<40} {'Test AUC (mean±std)':<25}")
    print("-" * 80)

    condition_desc = {
        'G': 'DFL pretrained + frozen',
        'H': 'DFL pretrained + unfrozen',
        'A': 'USSF pretrained + frozen',
    }

    for cond in ['G', 'H', 'A']:
        agg = results['aggregated'][cond]
        auc_str = f"{agg['test_auc_mean']:.4f} ± {agg['test_auc_std']:.4f}"
        print(f"{cond:<12} {condition_desc[cond]:<40} {auc_str:<25}")

    # Compare with baseline
    print("\n" + "-" * 80)
    print("Comparison with Phase 3 baselines:")
    print("  F (Majority baseline): 0.50 AUC")
    print("  D (Random init): 0.39 AUC")

    # Statistical comparison
    g_aucs = results['aggregated']['G']['aucs']
    a_aucs = results['aggregated']['A']['aucs']

    diff = np.mean(g_aucs) - np.mean(a_aucs)
    print(f"\nDFL vs USSF difference: {diff:+.4f}")

    if diff > 0:
        print("  → DFL open-play pretraining OUTPERFORMS USSF counterattack pretraining")
    else:
        print("  → USSF counterattack pretraining outperforms DFL open-play pretraining")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = RESULTS_DIR / f"phase5_comparison_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")

    # Summary JSON
    import json
    summary = {
        'timestamp': timestamp,
        'seeds': args.seeds,
        'aggregated': {
            cond: {
                'test_auc_mean': float(agg['test_auc_mean']),
                'test_auc_std': float(agg['test_auc_std']),
            }
            for cond, agg in results['aggregated'].items()
        },
        'dfl_vs_ussf_difference': float(diff)
    }
    summary_path = RESULTS_DIR / f"phase5_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Phase 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
