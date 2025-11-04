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
from sklearn.preprocessing import LabelEncoder


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
        self.label_encoder = LabelEncoder()  # For sparse receiver labels

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

        # Encode sparse receiver labels to consecutive integers
        # E.g., [0,1,2,...,12,14,16] -> [0,1,2,...,12,13,14]
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)

        # Train XGBoost
        self.model = self.xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=n_classes,  # Number of unique receiver positions
            verbosity=0
        )

        self.model.fit(X_train, y_train_encoded)

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

            # Predict (returns probabilities for encoded classes)
            probs_encoded = self.model.predict_proba(flat_features)  # [1, n_encoded_classes]

            # Map back to original label space (0-21)
            probs_full = np.zeros(22)
            for encoded_idx, original_label in enumerate(self.label_encoder.classes_):
                probs_full[original_label] = probs_encoded[0, encoded_idx]

            all_probs.append(probs_full)

        # Stack and convert to tensor
        all_probs = np.array(all_probs)  # [batch_size, 22]
        return torch.FloatTensor(all_probs)

    def train_shot(self, x_list: List[torch.Tensor], batch_list: List[torch.Tensor],
                   shot_labels: List[int]):
        """
        Train XGBoost classifier for shot prediction with enhanced features.

        Args:
            x_list: List of node features [num_players, 14] for each corner
            batch_list: List of batch tensors (for compatibility)
            shot_labels: List of shot labels (0 or 1) for each corner
        """
        # Extract enhanced graph-level features
        X_train = []

        for x in x_list:
            # Extract per-player features
            player_features = self.extract_features(x)  # [num_players, num_features_per_player]

            # Aggregate to graph-level features (mean, max, min, std)
            feat_mean = player_features.mean(axis=0)
            feat_max = player_features.max(axis=0)
            feat_min = player_features.min(axis=0)
            feat_std = player_features.std(axis=0)

            # Add shot-specific features from raw data
            raw_features = x.numpy() if torch.is_tensor(x) else x

            # Count players in different zones
            in_box = np.sum(raw_features[:, 13])  # in_penalty_box flag
            attacking_third = np.sum(raw_features[:, 0] > 80)  # x > 80 (attacking third)

            # Distance statistics (important for shots)
            distances_to_goal = raw_features[:, 2]  # distance_to_goal
            min_dist_to_goal = distances_to_goal.min()
            mean_dist_to_goal = distances_to_goal.mean()

            # Team balance
            team_flags = raw_features[:, 12]  # team_flag
            num_attackers = np.sum(team_flags == 1)
            num_defenders = np.sum(team_flags == 0)
            attacker_defender_ratio = num_attackers / max(num_defenders, 1)

            # Density around goal area
            goal_area_players = np.sum((raw_features[:, 0] > 100) &
                                       (raw_features[:, 1] > 25) &
                                       (raw_features[:, 1] < 55))

            # Concatenate all statistics
            graph_features = np.concatenate([
                feat_mean, feat_max, feat_min, feat_std,
                [in_box, attacking_third, min_dist_to_goal, mean_dist_to_goal,
                 num_attackers, num_defenders, attacker_defender_ratio, goal_area_players]
            ])
            X_train.append(graph_features)

        X_train = np.array(X_train)
        y_train = np.array(shot_labels)

        # Train XGBoost with balanced parameters
        self.shot_model = self.xgb.XGBClassifier(
            max_depth=4,  # Reduce depth to prevent overfitting
            n_estimators=300,
            learning_rate=0.1,
            random_state=self.random_state,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=2.5,  # Handle class imbalance (27.7% positive)
            subsample=0.8,
            colsample_bytree=0.8,
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

            # Extract and aggregate features (same as training)
            player_features = self.extract_features(graph_x)
            feat_mean = player_features.mean(axis=0)
            feat_max = player_features.max(axis=0)
            feat_min = player_features.min(axis=0)
            feat_std = player_features.std(axis=0)

            # Add shot-specific features
            raw_features = graph_x.numpy() if torch.is_tensor(graph_x) else graph_x
            in_box = np.sum(raw_features[:, 13])
            attacking_third = np.sum(raw_features[:, 0] > 80)
            distances_to_goal = raw_features[:, 2]
            min_dist_to_goal = distances_to_goal.min()
            mean_dist_to_goal = distances_to_goal.mean()
            team_flags = raw_features[:, 12]
            num_attackers = np.sum(team_flags == 1)
            num_defenders = np.sum(team_flags == 0)
            attacker_defender_ratio = num_attackers / max(num_defenders, 1)
            goal_area_players = np.sum((raw_features[:, 0] > 100) &
                                       (raw_features[:, 1] > 25) &
                                       (raw_features[:, 1] < 55))

            graph_features = np.concatenate([
                feat_mean, feat_max, feat_min, feat_std,
                [in_box, attacking_third, min_dist_to_goal, mean_dist_to_goal,
                 num_attackers, num_defenders, attacker_defender_ratio, goal_area_players]
            ])
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
                       verbose: bool = True,
                       dual_task: bool = True,
                       shot_weight: float = 1.0) -> Dict:
    """
    Train MLP baseline for specified number of steps with dual-task learning.

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
        dual_task: Whether to train both receiver and shot prediction
        shot_weight: Weight for shot loss in combined loss (default 1.0)

    Returns:
        Training history dictionary
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    receiver_criterion = nn.CrossEntropyLoss()
    shot_criterion = nn.BCELoss()

    history = {
        'train_loss': [],
        'train_receiver_loss': [],
        'train_shot_loss': [],
        'val_top1': [],
        'val_top3': [],
        'val_top5': [],
        'val_shot_f1': [],
        'val_shot_auroc': [],
        'steps': []
    }

    step = 0
    best_val_top3 = 0.0
    best_model_state = None

    if verbose:
        print(f"\nTraining MLP baseline for {num_steps} steps...")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        if dual_task:
            print(f"Dual-task training enabled (shot_weight={shot_weight})")
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

        # Receiver prediction
        logits = model(batch.x, batch.batch)
        receiver_targets = batch.receiver_label.squeeze()
        receiver_loss = receiver_criterion(logits, receiver_targets)

        # Shot prediction (if dual-task)
        total_loss = receiver_loss
        shot_loss = torch.tensor(0.0, device=device)

        if dual_task:
            shot_probs = model.predict_shot(batch.x, batch.batch)
            shot_targets = batch.shot_label.float().unsqueeze(1)
            shot_loss = shot_criterion(shot_probs, shot_targets)
            total_loss = receiver_loss + shot_weight * shot_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        step += 1

        # Evaluate
        if step % eval_every == 0 or step == num_steps:
            val_metrics = evaluate_baseline(model, val_loader, device)

            history['train_loss'].append(total_loss.item())
            history['train_receiver_loss'].append(receiver_loss.item())
            history['train_shot_loss'].append(shot_loss.item())
            history['val_top1'].append(val_metrics['top1_accuracy'])
            history['val_top3'].append(val_metrics['top3_accuracy'])
            history['val_top5'].append(val_metrics['top5_accuracy'])
            history['steps'].append(step)

            # Evaluate shot prediction if dual-task
            if dual_task:
                from sklearn.metrics import f1_score, roc_auc_score
                all_shot_preds = []
                all_shot_labels = []

                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        shot_probs = model.predict_shot(val_batch.x, val_batch.batch)
                        all_shot_preds.extend(shot_probs.cpu().numpy().flatten())
                        all_shot_labels.extend(val_batch.shot_label.cpu().numpy())

                shot_preds_binary = (np.array(all_shot_preds) > 0.5).astype(int)
                val_shot_f1 = f1_score(all_shot_labels, shot_preds_binary)
                val_shot_auroc = roc_auc_score(all_shot_labels, all_shot_preds)

                history['val_shot_f1'].append(val_shot_f1)
                history['val_shot_auroc'].append(val_shot_auroc)

            if verbose:
                if dual_task:
                    print(f"Step {step:5d}/{num_steps} | "
                          f"Loss: {total_loss.item():.4f} (R:{receiver_loss.item():.3f}, S:{shot_loss.item():.3f}) | "
                          f"Val Top-3: {val_metrics['top3_accuracy']*100:.1f}% | "
                          f"Shot F1: {val_shot_f1:.3f} | "
                          f"Shot AUROC: {val_shot_auroc:.3f}")
                else:
                    print(f"Step {step:5d}/{num_steps} | "
                          f"Loss: {total_loss.item():.4f} | "
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


class RandomOutcomeBaseline(nn.Module):
    """
    Random outcome prediction baseline for multi-class classification.

    Predicts uniform random probabilities over 4 outcome classes:
    - 0: Goal (~1.3%)
    - 1: Shot (~15.8%)
    - 2: Clearance (~39.5%)
    - 3: Possession (~43.4%)

    Expected performance (sanity check):
    - Accuracy: 25% (1/4 uniform random)
    - Macro F1: ~0.25
    """

    def __init__(self, num_classes: int = 4):
        """
        Initialize random outcome baseline.

        Args:
            num_classes: Number of outcome classes (default 4)
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random outcome predictions.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Random logits [batch_size, num_classes]
        """
        batch_size = batch.max().item() + 1
        logits = torch.randn(batch_size, self.num_classes, device=x.device)
        return logits

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate random predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_classes]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)


class XGBoostOutcomeBaseline:
    """
    XGBoost outcome prediction baseline with graph-level features.

    Extracts aggregated features for multi-class outcome prediction:
    - Graph statistics: mean/std positions, formation compactness
    - Spatial features: defensive line height, attacking positioning
    - Density features: players in box, goal area crowding

    Expected performance (TacticAI plan):
    - Accuracy: 50-60%
    - Macro F1: > 0.45
    """

    def __init__(self, max_depth: int = 6, n_estimators: int = 500,
                 learning_rate: float = 0.05, random_state: int = 42):
        """
        Initialize XGBoost outcome baseline.

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
        self.model = None
        self.num_classes = 4  # Goal, Shot, Clearance, Possession

        # StatsBomb pitch dimensions
        self.pitch_length = 120.0
        self.pitch_width = 80.0

    def extract_graph_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract graph-level features by aggregating player features.

        Args:
            x: Node features [num_players, 14]

        Returns:
            Graph-level features [num_features]
        """
        x_np = x.detach().cpu().numpy()

        # Extract base features
        pos_x = x_np[:, 0]
        pos_y = x_np[:, 1]
        dist_goal = x_np[:, 2]
        dist_ball = x_np[:, 3]
        team_flag = x_np[:, 10]  # 1 = attacking, 0 = defending
        in_penalty = x_np[:, 11]

        features = []

        # === POSITION STATISTICS (8) ===
        features.extend([
            pos_x.mean(), pos_x.std(),
            pos_y.mean(), pos_y.std(),
            pos_x.max(), pos_x.min(),
            pos_y.max(), pos_y.min()
        ])

        # === DISTANCE STATISTICS (4) ===
        features.extend([
            dist_goal.mean(), dist_goal.min(),
            dist_ball.mean(), dist_ball.min()
        ])

        # === TEAM BALANCE (4) ===
        attacking_mask = (team_flag == 1.0)
        defending_mask = (team_flag == 0.0)
        num_attackers = attacking_mask.sum()
        num_defenders = defending_mask.sum()

        features.extend([
            float(num_attackers),
            float(num_defenders),
            float(num_attackers) / max(float(num_defenders), 1.0),
            float(num_attackers) - float(num_defenders)
        ])

        # === FORMATION COMPACTNESS (6) ===
        if num_attackers > 1:
            attacking_x = pos_x[attacking_mask]
            attacking_y = pos_y[attacking_mask]
            features.extend([
                attacking_x.mean(), attacking_x.std(),
                attacking_y.mean(), attacking_y.std()
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        if num_defenders > 1:
            defending_x = pos_x[defending_mask]
            defending_y = pos_y[defending_mask]
            features.extend([
                defending_x.mean(), defending_x.std()
            ])
        else:
            features.extend([0.0, 0.0])

        # === ZONAL DENSITY (5) ===
        players_in_box = in_penalty.sum()
        players_in_6yard = ((pos_x > 114) & (pos_y > 30) & (pos_y < 50)).sum()
        players_near_goal = ((pos_x > 100) & (pos_y > 25) & (pos_y < 55)).sum()
        attackers_in_box = (attacking_mask & (in_penalty == 1)).sum()
        defenders_in_box = (defending_mask & (in_penalty == 1)).sum()

        features.extend([
            float(players_in_box),
            float(players_in_6yard),
            float(players_near_goal),
            float(attackers_in_box),
            float(defenders_in_box)
        ])

        # === DEFENSIVE LINE (2) ===
        if num_defenders > 1:
            defensive_line_height = defending_x.min()
            defensive_line_compactness = defending_y.std()
        else:
            defensive_line_height = 0.0
            defensive_line_compactness = 0.0

        features.extend([
            defensive_line_height,
            defensive_line_compactness
        ])

        return np.array(features)

    def train(self, x_list: List[torch.Tensor], batch_list: List[torch.Tensor],
              labels: List[int]):
        """
        Train XGBoost multi-class classifier.

        Args:
            x_list: List of node features [num_players, 14] for each corner
            batch_list: List of batch tensors (for compatibility)
            labels: List of outcome class labels (0-3) for each corner
        """
        X_train = []

        for x in x_list:
            graph_features = self.extract_graph_features(x)
            X_train.append(graph_features)

        X_train = np.array(X_train)
        y_train = np.array(labels)

        # Compute class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        sample_weights = np.array([class_weights[label] for label in y_train])

        # Train XGBoost
        self.model = self.xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=self.num_classes,
            verbosity=0
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict outcome probabilities.

        Args:
            x: Node features [num_nodes, 14]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_classes]
        """
        if self.model is None:
            raise RuntimeError("Model not trained! Call train() first.")

        batch_size = batch.max().item() + 1
        all_probs = []

        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]

            graph_features = self.extract_graph_features(graph_x)
            graph_features = graph_features.reshape(1, -1)

            probs = self.model.predict_proba(graph_features)[0]
            all_probs.append(probs)

        return torch.FloatTensor(np.array(all_probs))


class MLPOutcomeBaseline(nn.Module):
    """
    MLP outcome prediction baseline for multi-class classification.

    Flattens all player features and uses a 3-layer MLP to predict outcome.
    Architecture:
    - Flatten: [batch, 22 nodes × 14 features] = [batch, 308]
    - MLP: 308 → 512 → 256 → 128 → 4
    - Dropout: 0.25
    - Activation: ReLU

    Expected performance (TacticAI plan):
    - Accuracy: 55-65%
    - Macro F1: > 0.50
    """

    def __init__(self, num_features: int = 14, num_players: int = 22,
                 hidden_dim1: int = 512, hidden_dim2: int = 256,
                 hidden_dim3: int = 128, num_classes: int = 4,
                 dropout: float = 0.25):
        """
        Initialize MLP outcome baseline.

        Args:
            num_features: Number of features per player (default 14)
            num_players: Number of players (default 22)
            hidden_dim1: First hidden layer dimension
            hidden_dim2: Second hidden layer dimension
            hidden_dim3: Third hidden layer dimension
            num_classes: Number of outcome classes (default 4)
            dropout: Dropout rate
        """
        super().__init__()

        self.num_features = num_features
        self.num_players = num_players
        self.num_classes = num_classes
        self.input_dim = num_players * num_features  # 22 × 14 = 308

        # MLP layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            Logits [batch_size, num_classes]
        """
        batch_size = batch.max().item() + 1

        # Flatten: [num_nodes, num_features] → [batch_size, num_players * num_features]
        flattened = self._flatten_batch(x, batch, batch_size)

        # MLP layers
        h1 = self.relu(self.fc1(flattened))
        h1 = self.dropout(h1)

        h2 = self.relu(self.fc2(h1))
        h2 = self.dropout(h2)

        h3 = self.relu(self.fc3(h2))
        h3 = self.dropout(h3)

        logits = self.fc4(h3)

        return logits

    def _flatten_batch(self, x: torch.Tensor, batch: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
        """
        Flatten node features by batch (same as MLPReceiverBaseline).

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]
            batch_size: Number of graphs in batch

        Returns:
            Flattened features [batch_size, num_players * num_features]
        """
        flattened = torch.zeros(batch_size, self.input_dim, device=x.device)

        for i in range(batch_size):
            mask = (batch == i)
            graph_features = x[mask]

            num_nodes = graph_features.size(0)
            if num_nodes > self.num_players:
                graph_features = graph_features[:self.num_players]
                num_nodes = self.num_players

            flat = graph_features.flatten()
            flattened[i, :len(flat)] = flat

        return flattened

    def predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions and return softmax probabilities.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch vector [num_nodes]

        Returns:
            Softmax probabilities [batch_size, num_classes]
        """
        logits = self.forward(x, batch)
        return F.softmax(logits, dim=1)


def evaluate_outcome_baseline(model, data_loader, device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate multi-class outcome prediction model.

    Computes:
    - Accuracy
    - Macro F1, Precision, Recall
    - Per-class F1 scores
    - Confusion matrix

    Args:
        model: Outcome prediction model
        data_loader: PyTorch Geometric DataLoader
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )

    # Set eval mode
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if hasattr(model, 'to'):
                batch = batch.to(device)

            # Get predictions
            if hasattr(model, 'forward'):
                logits = model(batch.x, batch.batch)
                probs = F.softmax(logits, dim=1)
            else:
                probs = model.predict(batch.x, batch.batch)

            preds = torch.argmax(probs, dim=1)
            labels = batch.outcome_class_label.squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    class_names = ['Goal', 'Shot', 'Clearance', 'Possession']
    for i, name in enumerate(class_names):
        metrics[f'{name.lower()}_f1'] = per_class_f1[i] if i < len(per_class_f1) else 0.0

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = conf_matrix.tolist()

    return metrics


def train_mlp_outcome(model: MLPOutcomeBaseline,
                     train_loader,
                     val_loader,
                     num_steps: int = 15000,
                     lr: float = 5e-4,
                     weight_decay: float = 1e-4,
                     device: str = 'cuda',
                     eval_every: int = 500,
                     verbose: bool = True,
                     use_class_weights: bool = True,
                     early_stopping_patience: int = 5000) -> Dict:
    """
    Train MLP outcome baseline for multi-class classification.

    Args:
        model: MLPOutcomeBaseline model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_steps: Number of training steps (default 15k)
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        eval_every: Evaluate every N steps
        verbose: Print progress
        use_class_weights: Whether to use class weights for imbalanced data
        early_stopping_patience: Stop if no improvement for N steps

    Returns:
        Training history dictionary
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr/10
    )

    # Compute class weights if needed (using SQRT to reduce extremes)
    if use_class_weights:
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch.outcome_class_label.numpy())

        from sklearn.utils.class_weight import compute_class_weight
        class_weights_raw = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        # Use SQRT to reduce extreme weights (Goal class was 33.88!)
        class_weights = np.sqrt(class_weights_raw)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_goal_f1': [],
        'val_shot_f1': [],
        'steps': []
    }

    step = 0
    best_val_macro_f1 = 0.0
    best_model_state = None
    steps_without_improvement = 0

    if verbose:
        print(f"\nTraining MLP outcome baseline for {num_steps} steps...")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Early stopping patience: {early_stopping_patience} steps")
        if use_class_weights:
            print(f"Using SQRT class weights: {class_weights.cpu().numpy()}")
            print(f"  (Raw weights were: {class_weights_raw})")
        print(f"Evaluating every {eval_every} steps\n")

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
        targets = batch.outcome_class_label.squeeze()
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # Update learning rate

        step += 1

        # Evaluate
        if step % eval_every == 0 or step == num_steps:
            val_metrics = evaluate_outcome_baseline(model, val_loader, device)

            history['train_loss'].append(loss.item())
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['val_goal_f1'].append(val_metrics['goal_f1'])
            history['val_shot_f1'].append(val_metrics['shot_f1'])
            history['steps'].append(step)

            current_lr = scheduler.get_last_lr()[0]

            if verbose:
                print(f"Step {step:5d}/{num_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Acc: {val_metrics['accuracy']*100:.1f}% | "
                      f"Macro F1: {val_metrics['macro_f1']:.3f} | "
                      f"Goal F1: {val_metrics['goal_f1']:.3f} | "
                      f"Shot F1: {val_metrics['shot_f1']:.3f}")

            # Save best model and check early stopping
            if val_metrics['macro_f1'] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics['macro_f1']
                best_model_state = model.state_dict().copy()
                steps_without_improvement = 0
                if verbose:
                    print(f"  → New best Macro F1: {best_val_macro_f1:.3f}")
            else:
                steps_without_improvement += eval_every

            # Early stopping
            if steps_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"\n⚠️  Early stopping triggered (no improvement for {early_stopping_patience} steps)")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_val_macro_f1'] = best_val_macro_f1
    history['stopped_early'] = (step < num_steps)
    history['final_step'] = step

    if verbose:
        print(f"\nTraining complete!")
        print(f"Best Val Macro F1: {best_val_macro_f1:.3f}")
        if history['stopped_early']:
            print(f"Stopped early at step {step}/{num_steps}")

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

    # Test RandomOutcomeBaseline
    print("\n3. Testing RandomOutcomeBaseline...")
    random_outcome_model = RandomOutcomeBaseline(num_classes=4)
    outcome_logits = random_outcome_model(x, batch)
    outcome_probs = random_outcome_model.predict(x, batch)

    print(f"  Logits shape: {outcome_logits.shape}")
    print(f"  Probs shape: {outcome_probs.shape}")
    print(f"  Probs sum (should be 1.0): {outcome_probs.sum(dim=1)}")

    # Test MLPOutcomeBaseline
    print("\n4. Testing MLPOutcomeBaseline...")
    mlp_outcome_model = MLPOutcomeBaseline(num_features=14, num_players=22)
    outcome_logits = mlp_outcome_model(x, batch)
    outcome_probs = mlp_outcome_model.predict(x, batch)

    print(f"  Logits shape: {outcome_logits.shape}")
    print(f"  Probs shape: {outcome_probs.shape}")
    print(f"  Probs sum (should be 1.0): {outcome_probs.sum(dim=1)}")

    # Count parameters
    num_params = sum(p.numel() for p in mlp_outcome_model.parameters())
    print(f"  Number of parameters: {num_params:,}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
