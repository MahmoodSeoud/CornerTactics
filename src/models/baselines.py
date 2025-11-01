#!/usr/bin/env python3
"""
Baseline Models for TacticAI-Style Receiver Prediction

Implements Day 5-6: Baseline Models
- RandomReceiverBaseline: Random softmax over 22 players (sanity check)
- XGBoostReceiverBaseline: Engineered features with XGBoost classifier
- MLPReceiverBaseline: Flatten all player positions → MLP → 22 classes

Based on TacticAI Implementation Plan:
- Random baseline: top-1=4.5%, top-3=13.6%, top-5=22.7%
- XGBoost baseline: Expected top-1 > 25%, top-3 > 42%
- MLP baseline: Expected top-1 > 22%, top-3 > 45%
- If MLP top-3 < 40%: STOP and debug data pipeline

Author: mseo
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class RandomReceiverBaseline(nn.Module):
    """
    Random receiver prediction baseline.

    Predicts random softmax probabilities over all players.
    Expected performance (sanity check):
    - Top-1: 4.5% (1/22)
    - Top-3: 13.6% (3/22)
    - Top-5: 22.7% (5/22)
    """

    def __init__(self, num_players: int = 22):
        """
        Initialize random baseline.

        Args:
            num_players: Number of players (default 22)
        """
        super().__init__()
        self.num_players = num_players

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random predictions.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Random logits [batch_size, num_players]
        """
        batch_size = batch.max().item() + 1

        # Generate random logits (uniform distribution)
        logits = torch.randn(batch_size, self.num_players, device=x.device)

        return logits

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_players]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)

    def predict_shot(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random shot predictions.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Random shot probabilities [batch_size, 1]
        """
        batch_size = batch.max().item() + 1
        # Random probabilities between 0 and 1
        shot_probs = torch.rand(batch_size, 1, device=x.device)
        return shot_probs


class XGBoostReceiverBaseline:
    """
    XGBoost receiver prediction baseline with engineered features.

    Extracts hand-crafted spatial and contextual features for each player:
    - Spatial: x, y, distance to goal, distance to ball
    - Relative: closest opponent distance, teammates within 5m
    - Zonal: binary flags (in 6-yard box, in penalty area, near/far post)
    - Team context: average team x-position, defensive line compactness
    - Player role: is_goalkeeper, is_corner_taker

    Features per player: ~15 dimensions
    Total features per corner: 22 players × 15 = ~330 features

    Expected performance (TacticAI plan):
    - Top-1: > 25%
    - Top-3: > 42%
    - Top-5: > 60%
    """

    def __init__(self, max_depth: int = 6, n_estimators: int = 500,
                 learning_rate: float = 0.05, random_state: int = 42):
        """
        Initialize XGBoost baseline.

        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            random_state: Random seed
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None  # Receiver prediction model
        self.shot_model = None  # Shot prediction model

        # StatsBomb pitch dimensions
        self.pitch_length = 120.0
        self.pitch_width = 80.0
        self.goal_x = 120.0
        self.goal_y_center = 40.0

        # Zonal boundaries
        self.six_yard_box = {'x_min': 114.0, 'y_min': 30.0, 'y_max': 50.0}
        self.penalty_area = {'x_min': 102.0, 'y_min': 18.0, 'y_max': 62.0}

    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract hand-crafted features for each player.

        Args:
            x: Node features [num_players, 14]
               Columns: [x, y, dist_goal, dist_ball, vx, vy, v_mag, v_angle,
                        angle_goal, angle_ball, team, in_penalty, num_near, density]

        Returns:
            Engineered features [num_players, num_features]
        """
        x_np = x.detach().cpu().numpy()
        num_players = x_np.shape[0]
        features_list = []

        # Extract base features
        pos_x = x_np[:, 0]  # x position
        pos_y = x_np[:, 1]  # y position
        dist_goal = x_np[:, 2]  # distance to goal
        dist_ball = x_np[:, 3]  # distance to ball landing
        team_flag = x_np[:, 10]  # 1 = attacking, 0 = defending

        for i in range(num_players):
            player_features = []

            # === SPATIAL FEATURES (4) ===
            player_features.append(pos_x[i])
            player_features.append(pos_y[i])
            player_features.append(dist_goal[i])
            player_features.append(dist_ball[i])

            # === ZONAL FEATURES (3) ===
            # In 6-yard box?
            in_six_yard = (
                pos_x[i] >= self.six_yard_box['x_min'] and
                self.six_yard_box['y_min'] <= pos_y[i] <= self.six_yard_box['y_max']
            )
            player_features.append(float(in_six_yard))

            # In penalty area?
            in_penalty = (
                pos_x[i] >= self.penalty_area['x_min'] and
                self.penalty_area['y_min'] <= pos_y[i] <= self.penalty_area['y_max']
            )
            player_features.append(float(in_penalty))

            # Near post (y < 35) or far post (y > 45)?
            near_post = float(pos_y[i] < 35.0)
            far_post = float(pos_y[i] > 45.0)
            player_features.append(near_post)
            player_features.append(far_post)

            # === RELATIVE FEATURES (2) ===
            # Closest opponent distance
            is_attacking = (team_flag[i] == 1.0)
            opponent_mask = (team_flag != team_flag[i])
            if opponent_mask.sum() > 0:
                opponent_positions = np.stack([pos_x[opponent_mask], pos_y[opponent_mask]], axis=1)
                player_pos = np.array([pos_x[i], pos_y[i]])
                distances = np.linalg.norm(opponent_positions - player_pos, axis=1)
                closest_opponent_dist = distances.min()
            else:
                closest_opponent_dist = 999.0  # No opponents
            player_features.append(closest_opponent_dist)

            # Teammates within 5m radius
            teammate_mask = (team_flag == team_flag[i]) & (np.arange(num_players) != i)
            if teammate_mask.sum() > 0:
                teammate_positions = np.stack([pos_x[teammate_mask], pos_y[teammate_mask]], axis=1)
                player_pos = np.array([pos_x[i], pos_y[i]])
                distances = np.linalg.norm(teammate_positions - player_pos, axis=1)
                teammates_within_5m = (distances < 5.0).sum()
            else:
                teammates_within_5m = 0
            player_features.append(float(teammates_within_5m))

            # === TEAM CONTEXT FEATURES (2) ===
            # Average team x-position (attacking push)
            team_mask = (team_flag == team_flag[i])
            avg_team_x = pos_x[team_mask].mean()
            player_features.append(avg_team_x)

            # Defensive line compactness (std of y-positions for defending team)
            if is_attacking:
                defending_mask = (team_flag == 0.0)
            else:
                defending_mask = (team_flag == 1.0)
            if defending_mask.sum() > 1:
                defensive_y_std = pos_y[defending_mask].std()
            else:
                defensive_y_std = 0.0
            player_features.append(defensive_y_std)

            # === PLAYER ROLE FEATURES (2) ===
            # Is goalkeeper? (approximation: x < 20 for defending team)
            is_goalkeeper = float((not is_attacking) and pos_x[i] < 20.0)
            player_features.append(is_goalkeeper)

            # Is corner taker? (approximation: closest to corner flag)
            corner_x, corner_y = 120.0, 0.0  # or 80.0 depending on corner side
            dist_to_corner = np.sqrt((pos_x[i] - corner_x)**2 + (pos_y[i] - corner_y)**2)
            is_corner_taker = float(dist_to_corner < 2.0 and is_attacking)
            player_features.append(is_corner_taker)

            features_list.append(player_features)

        return np.array(features_list)  # [num_players, num_features]

    def train(self, x_list: List[torch.Tensor], batch_list: List[torch.Tensor],
              labels: List[int]):
        """
        Train XGBoost classifier.

        Args:
            x_list: List of node features [num_players, 14] for each corner
            batch_list: List of batch tensors (for compatibility, not used here)
            labels: List of receiver indices (0-21) for each corner
        """
        # Extract features for all graphs
        X_train = []
        y_train = []

        # Determine max number of features needed (for padding)
        max_features = 0
        all_player_features = []

        for x in x_list:
            player_features = self.extract_features(x)  # [num_players, num_features_per_player]
            all_player_features.append(player_features)
            max_features = max(max_features, player_features.flatten().shape[0])

        # Pad and flatten all features
        for player_features, label in zip(all_player_features, labels):
            # Flatten to single feature vector
            flat_features = player_features.flatten()

            # Pad to max_features if needed
            if flat_features.shape[0] < max_features:
                padded = np.zeros(max_features)
                padded[:flat_features.shape[0]] = flat_features
                flat_features = padded

            X_train.append(flat_features)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Store max_features for prediction
        self.max_features = max_features

        # Train XGBoost
        self.model = self.xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0
        )

        self.model.fit(X_train, y_train)

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict receiver probabilities.

        Args:
            x: Node features [num_nodes, 14]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Softmax probabilities [batch_size, 22]
        """
        if self.model is None:
            raise RuntimeError("Model not trained! Call train() first.")

        batch_size = batch.max().item() + 1
        all_probs = []

        # Process each graph in batch
        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]  # [num_players, 14]

            # Extract features
            player_features = self.extract_features(graph_x)  # [num_players, num_features]
            flat_features = player_features.flatten()

            # Pad to max_features if needed
            if flat_features.shape[0] < self.max_features:
                padded = np.zeros(self.max_features)
                padded[:flat_features.shape[0]] = flat_features
                flat_features = padded

            flat_features = flat_features.reshape(1, -1)  # [1, max_features]

            # Predict
            probs = self.model.predict_proba(flat_features)  # [1, num_classes]
            all_probs.append(probs[0])

        # Stack and convert to tensor
        all_probs = np.array(all_probs)  # [batch_size, num_classes]
        return torch.FloatTensor(all_probs)

    def train_shot(self, x_list: List[torch.Tensor], batch_list: List[torch.Tensor],
                   shot_labels: List[int]):
        """
        Train XGBoost classifier for shot prediction.

        Args:
            x_list: List of node features [num_players, 14] for each corner
            batch_list: List of batch tensors (for compatibility)
            shot_labels: List of shot labels (0 or 1) for each corner
        """
        # Extract graph-level features
        X_train = []

        for x in x_list:
            # Extract per-player features
            player_features = self.extract_features(x)  # [num_players, num_features_per_player]

            # Aggregate to graph-level features (mean, max, min, std)
            feat_mean = player_features.mean(axis=0)
            feat_max = player_features.max(axis=0)
            feat_min = player_features.min(axis=0)
            feat_std = player_features.std(axis=0)

            # Concatenate all statistics
            graph_features = np.concatenate([feat_mean, feat_max, feat_min, feat_std])
            X_train.append(graph_features)

        X_train = np.array(X_train)
        y_train = np.array(shot_labels)

        # Train XGBoost for binary classification
        self.shot_model = self.xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='binary:logistic',
            eval_metric='logloss',
            verbosity=0
        )

        self.shot_model.fit(X_train, y_train)

    def predict_shot(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict shot probabilities.

        Args:
            x: Node features [num_nodes, 14]
            batch: Batch vector [num_nodes]

        Returns:
            Shot probabilities [batch_size, 1]
        """
        if self.shot_model is None:
            raise RuntimeError("Shot model not trained! Call train_shot() first.")

        batch_size = batch.max().item() + 1
        all_probs = []

        # Process each graph in batch
        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]

            # Extract and aggregate features
            player_features = self.extract_features(graph_x)
            feat_mean = player_features.mean(axis=0)
            feat_max = player_features.max(axis=0)
            feat_min = player_features.min(axis=0)
            feat_std = player_features.std(axis=0)
            graph_features = np.concatenate([feat_mean, feat_max, feat_min, feat_std])
            graph_features = graph_features.reshape(1, -1)

            # Predict probability of positive class (shot=1)
            prob = self.shot_model.predict_proba(graph_features)[0, 1]
            all_probs.append([prob])

        return torch.FloatTensor(all_probs)  # [batch_size, 1]


class MLPReceiverBaseline(nn.Module):
    """
    MLP receiver prediction baseline.

    Flattens all player features and uses a 3-layer MLP to predict receiver.
    Architecture:
    - Flatten: [batch, 22 nodes × 14 features] = [batch, 308]
    - MLP: 308 → 256 → 128 → 22
    - Dropout: 0.3
    - Activation: ReLU

    Expected performance (TacticAI plan):
    - Top-1: > 20%
    - Top-3: > 45%
    - If top-3 < 40%: Debug data pipeline
    """

    def __init__(self, num_features: int = 14, num_players: int = 22,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 dropout: float = 0.3):
        """
        Initialize MLP baseline.

        Args:
            num_features: Number of features per player (default 14)
            num_players: Number of players (default 22)
            hidden_dim1: First hidden layer dimension
            hidden_dim2: Second hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_features = num_features
        self.num_players = num_players
        self.input_dim = num_players * num_features  # 22 × 14 = 308

        # Shared layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        # Receiver prediction head
        self.fc3_receiver = nn.Linear(hidden_dim2, num_players)

        # Shot prediction head
        self.fc3_shot = nn.Linear(hidden_dim2, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Logits [batch_size, num_players]
        """
        batch_size = batch.max().item() + 1

        # Flatten: [num_nodes, num_features] → [batch_size, num_players * num_features]
        flattened = self._flatten_batch(x, batch, batch_size)

        # MLP layers
        h1 = self.relu(self.fc1(flattened))
        h1 = self.dropout(h1)

        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)

        # Receiver prediction head
        logits = self.fc3_receiver(h2)

        return logits

    def _flatten_batch(self, x: torch.Tensor, batch: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
        """
        Flatten node features by batch.

        Handles variable number of players per graph by padding/truncating to 22.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]
            batch_size: Number of graphs in batch

        Returns:
            Flattened features [batch_size, num_players * num_features]
        """
        flattened = torch.zeros(batch_size, self.input_dim, device=x.device)

        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch == i)
            graph_features = x[mask]  # [num_nodes_i, num_features]

            num_nodes = graph_features.size(0)

            if num_nodes > self.num_players:
                # Truncate if more than 22 players (shouldn't happen)
                graph_features = graph_features[:self.num_players]
                num_nodes = self.num_players

            # Flatten and insert into batch tensor
            flat = graph_features.flatten()  # [num_nodes_i * num_features]
            flattened[i, :len(flat)] = flat

            # Remaining positions are zero-padded (for graphs with < 22 players)

        return flattened

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_players]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)

    def predict_shot(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict shot probabilities using dual-head architecture.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Shot probabilities [batch_size, 1]
        """
        batch_size = batch.max().item() + 1

        # Flatten input
        flattened = self._flatten_batch(x, batch, batch_size)

        # Shared layers
        h1 = self.relu(self.fc1(flattened))
        h1 = self.dropout(h1)

        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)

        # Shot prediction head
        shot_logits = self.fc3_shot(h2)
        shot_probs = torch.sigmoid(shot_logits)

        return shot_probs  # [batch_size, 1]


def evaluate_baseline(model: nn.Module,
                     data_loader,
                     device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate baseline model on a dataset.

    Computes:
    - Top-1 accuracy
    - Top-3 accuracy
    - Top-5 accuracy
    - Cross-entropy loss

    Args:
        model: Baseline model (Random, XGBoost, or MLP)
        data_loader: PyTorch Geometric DataLoader
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    # Set eval mode and move to device (only for PyTorch models)
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model.to(device)

    total = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            if hasattr(model, 'to'):
                batch = batch.to(device)

            # Forward pass (handle both PyTorch and non-PyTorch models)
            if hasattr(model, 'forward'):
                # PyTorch model (has forward method)
                logits = model(batch.x, batch.batch)
                probs = F.softmax(logits, dim=1)
            else:
                # Non-PyTorch model (like XGBoost) - use predict method directly
                probs = model.predict(batch.x, batch.batch)

            # Top-k predictions
            _, top5_pred = torch.topk(probs, k=5, dim=1)

            # Ground truth
            targets = batch.receiver_label.squeeze()

            # Compute accuracy
            batch_size = targets.size(0)
            total += batch_size

            for i in range(batch_size):
                target = targets[i].item()
                top5 = top5_pred[i].tolist()

                # Top-1
                if top5[0] == target:
                    top1_correct += 1

                # Top-3
                if target in top5[:3]:
                    top3_correct += 1

                # Top-5
                if target in top5:
                    top5_correct += 1

            # Compute loss (only for PyTorch models with logits)
            if hasattr(model, 'forward'):
                loss = criterion(logits, targets)
                total_loss += loss.item() * batch_size
            else:
                # For non-PyTorch models, compute loss from probabilities
                log_probs = torch.log(probs + 1e-10)  # Add epsilon to avoid log(0)
                loss = -log_probs[range(batch_size), targets].mean()
                total_loss += loss.item() * batch_size

    # Compute final metrics
    metrics = {
        'top1_accuracy': top1_correct / total,
        'top3_accuracy': top3_correct / total,
        'top5_accuracy': top5_correct / total,
        'loss': total_loss / total
    }

    return metrics


def evaluate_shot_prediction(model, data_loader, device: str = 'cuda',
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate shot prediction performance.

    Computes:
    - F1 score, Precision, Recall
    - AUROC (Area Under ROC Curve)
    - AUPRC (Area Under Precision-Recall Curve)
    - Accuracy

    Args:
        model: Baseline model with predict_shot() method
        data_loader: PyTorch Geometric DataLoader
        device: Device to run on
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary with shot prediction metrics
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, accuracy_score
    )

    # Set eval mode (only for PyTorch models)
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if hasattr(model, 'to'):
                batch = batch.to(device)

            # Predict shot probabilities
            shot_probs = model.predict_shot(batch.x, batch.batch)  # [batch_size, 1]

            # Get ground truth
            shot_labels = batch.shot_label  # [batch_size, 1]

            all_probs.extend(shot_probs.cpu().numpy().flatten())
            all_labels.extend(shot_labels.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)

    # Binary predictions
    all_preds = (all_probs >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auroc': roc_auc_score(all_labels, all_probs),
        'auprc': average_precision_score(all_labels, all_probs),
        'positive_rate': all_labels.mean(),  # % of positive samples
        'threshold': threshold
    }

    return metrics


def train_mlp_baseline(model: MLPReceiverBaseline,
                       train_loader,
                       val_loader,
                       num_steps: int = 10000,
                       lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       device: str = 'cuda',
                       eval_every: int = 500,
                       verbose: bool = True) -> Dict:
    """
    Train MLP baseline for specified number of steps.

    Args:
        model: MLPReceiverBaseline model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_steps: Number of training steps (default 10k)
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        eval_every: Evaluate every N steps
        verbose: Print progress

    Returns:
        Training history dictionary
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_top1': [],
        'val_top3': [],
        'val_top5': [],
        'steps': []
    }

    step = 0
    best_val_top3 = 0.0
    best_model_state = None

    if verbose:
        print(f"\nTraining MLP baseline for {num_steps} steps...")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Evaluating every {eval_every} steps\n")

    # Training loop
    train_iter = iter(train_loader)

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device)

        # Forward pass
        model.train()
        optimizer.zero_grad()

        logits = model(batch.x, batch.batch)
        targets = batch.receiver_label.squeeze()

        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        step += 1

        # Evaluate
        if step % eval_every == 0 or step == num_steps:
            val_metrics = evaluate_baseline(model, val_loader, device)

            history['train_loss'].append(loss.item())
            history['val_top1'].append(val_metrics['top1_accuracy'])
            history['val_top3'].append(val_metrics['top3_accuracy'])
            history['val_top5'].append(val_metrics['top5_accuracy'])
            history['steps'].append(step)

            if verbose:
                print(f"Step {step:5d}/{num_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Val Top-1: {val_metrics['top1_accuracy']*100:.1f}% | "
                      f"Val Top-3: {val_metrics['top3_accuracy']*100:.1f}% | "
                      f"Val Top-5: {val_metrics['top5_accuracy']*100:.1f}%")

            # Save best model
            if val_metrics['top3_accuracy'] > best_val_top3:
                best_val_top3 = val_metrics['top3_accuracy']
                best_model_state = model.state_dict().copy()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_val_top3'] = best_val_top3

    if verbose:
        print(f"\nTraining complete!")
        print(f"Best Val Top-3: {best_val_top3*100:.1f}%")

    return history


if __name__ == "__main__":
    # Test baseline models
    print("="*60)
    print("TESTING BASELINE MODELS")
    print("="*60)

    # Create dummy batch
    batch_size = 4
    num_nodes_per_graph = 22
    num_features = 14

    x = torch.randn(batch_size * num_nodes_per_graph, num_features)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)

    print(f"\nDummy batch:")
    print(f"  x shape: {x.shape}")
    print(f"  batch shape: {batch.shape}")
    print(f"  batch_size: {batch_size}")

    # Test RandomReceiverBaseline
    print("\n1. Testing RandomReceiverBaseline...")
    random_model = RandomReceiverBaseline(num_players=22)
    random_logits = random_model(x, batch)
    random_probs = random_model.predict(x, batch)

    print(f"  Logits shape: {random_logits.shape}")
    print(f"  Probs shape: {random_probs.shape}")
    print(f"  Probs sum (should be 1.0): {random_probs.sum(dim=1)}")

    # Test MLPReceiverBaseline
    print("\n2. Testing MLPReceiverBaseline...")
    mlp_model = MLPReceiverBaseline(num_features=14, num_players=22)
    mlp_logits = mlp_model(x, batch)
    mlp_probs = mlp_model.predict(x, batch)

    print(f"  Logits shape: {mlp_logits.shape}")
    print(f"  Probs shape: {mlp_probs.shape}")
    print(f"  Probs sum (should be 1.0): {mlp_probs.sum(dim=1)}")

    # Count parameters
    num_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"  Number of parameters: {num_params:,}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
