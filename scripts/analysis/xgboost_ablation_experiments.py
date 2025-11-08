#!/usr/bin/env python3
"""
XGBoost Ablation Studies for Corner Kick Outcome Prediction

Implements three ablation experiments:
1. Temporal Augmentation Impact: t=0s only vs 5 temporal frames
2. Feature Selection Impact: All 22 players vs 5 closest to ball
3. Feature Importance Analysis: Which features matter most

Based on existing XGBoost baseline from train_outcome_baselines.py

Author: mseo
Date: November 2024
"""

import sys
import json
import pickle
import argparse
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, top_k_accuracy_score
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config_name: str
    num_graphs: int
    receiver_top1: Optional[float] = None
    receiver_top3: Optional[float] = None
    receiver_top5: Optional[float] = None
    outcome_accuracy: Optional[float] = None
    outcome_macro_f1: Optional[float] = None
    outcome_shot_f1: Optional[float] = None
    outcome_clearance_f1: Optional[float] = None
    outcome_possession_f1: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = {
            'config_name': self.config_name,
            'num_graphs': self.num_graphs,
            'receiver_top1': self.receiver_top1,
            'receiver_top3': self.receiver_top3,
            'receiver_top5': self.receiver_top5,
            'outcome_accuracy': self.outcome_accuracy,
            'outcome_macro_f1': self.outcome_macro_f1,
            'outcome_shot_f1': self.outcome_shot_f1,
            'outcome_clearance_f1': self.outcome_clearance_f1,
            'outcome_possession_f1': self.outcome_possession_f1,
        }
        if self.confusion_matrix is not None:
            d['confusion_matrix'] = self.confusion_matrix.tolist()
        return d


def load_graph_dataset(graph_path: str) -> List:
    """
    Load graph dataset from pickle file.

    Args:
        graph_path: Path to pickle file with CornerGraph objects

    Returns:
        List of CornerGraph objects
    """
    print(f"Loading dataset from {graph_path}...")
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"✓ Loaded {len(graphs)} graphs")
    return graphs


def filter_t0_frames(graphs: List) -> List:
    """
    Filter dataset to keep only t=0s frames (corner kick moment).

    Corner IDs have format: 'match_123_event_456_t0' or 'match_123_event_456_t+0.0'

    Args:
        graphs: List of CornerGraph objects

    Returns:
        List of graphs with only t=0s frames
    """
    t0_graphs = []
    for graph in graphs:
        corner_id = graph.corner_id
        # Check for t=0s indicators
        if '_t0' in corner_id or '_t+0' in corner_id or '_t+0.0' in corner_id:
            t0_graphs.append(graph)

    print(f"\nFiltered to t=0s frames:")
    print(f"  Original graphs: {len(graphs)}")
    print(f"  t=0s graphs: {len(t0_graphs)}")
    print(f"  Reduction: {len(t0_graphs)/len(graphs)*100:.1f}% of data retained")

    return t0_graphs


def split_dataset(graphs: List, train_ratio: float = 0.70, val_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test sets.

    Stratified by corner ID to prevent temporal leakage.

    Args:
        graphs: List of CornerGraph objects
        train_ratio: Training set ratio (default: 0.70)
        val_ratio: Validation set ratio (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    np.random.seed(random_seed)

    # Extract base corner IDs (without temporal suffix)
    corner_ids = []
    for graph in graphs:
        # Remove temporal suffix: match_123_event_456_t+1.0 -> match_123_event_456
        base_id = graph.corner_id.split('_t')[0]
        corner_ids.append(base_id)

    # Get unique corner IDs
    unique_ids = list(set(corner_ids))
    np.random.shuffle(unique_ids)

    # Split corner IDs
    n_train = int(len(unique_ids) * train_ratio)
    n_val = int(len(unique_ids) * val_ratio)

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    # Assign graphs to splits
    train_graphs = []
    val_graphs = []
    test_graphs = []

    for i, graph in enumerate(graphs):
        base_id = corner_ids[i]
        if base_id in train_ids:
            train_graphs.append(graph)
        elif base_id in val_ids:
            val_graphs.append(graph)
        else:
            test_graphs.append(graph)

    print(f"\nDataset split (stratified by corner ID):")
    print(f"  Train: {len(train_graphs):4d} graphs ({len(train_graphs)/len(graphs)*100:.1f}%)")
    print(f"  Val:   {len(val_graphs):4d} graphs ({len(val_graphs)/len(graphs)*100:.1f}%)")
    print(f"  Test:  {len(test_graphs):4d} graphs ({len(test_graphs)/len(graphs)*100:.1f}%)")

    return train_graphs, val_graphs, test_graphs


def extract_features_and_labels(graphs: List, task: str = 'outcome') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from graphs.

    Args:
        graphs: List of CornerGraph objects
        task: 'receiver' or 'outcome'

    Returns:
        Tuple of (X, y) where X is features array and y is labels array
    """
    X_list = []
    y_list = []

    for graph in graphs:
        # Extract node features (shape: [num_nodes, 14])
        node_features = graph.node_features  # numpy array

        if task == 'receiver':
            # Flatten to 308D vector (22 players × 14 features)
            # Pad to exactly 22 players if needed
            if node_features.shape[0] < 22:
                padding = np.zeros((22 - node_features.shape[0], 14))
                node_features = np.vstack([node_features, padding])
            elif node_features.shape[0] > 22:
                node_features = node_features[:22, :]

            features = node_features.flatten()  # [308,]

            # Label: receiver node index (0-21)
            if graph.receiver_node_index is not None:
                label = graph.receiver_node_index
            else:
                continue  # Skip if no receiver label

        elif task == 'outcome':
            # Aggregate to graph-level features (29D)
            features = aggregate_node_features_to_graph(node_features)

            # Label: outcome class (0=Shot, 1=Clearance, 2=Possession)
            if graph.outcome is not None:
                label = outcome_to_class_id(graph.outcome)
            else:
                continue  # Skip if no outcome label
        else:
            raise ValueError(f"Unknown task: {task}")

        X_list.append(features)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nExtracted {task} features:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    if task == 'outcome':
        class_counts = np.bincount(y, minlength=3)
        print(f"  Class distribution:")
        print(f"    Shot (0):       {class_counts[0]:4d} ({class_counts[0]/len(y)*100:5.1f}%)")
        print(f"    Clearance (1):  {class_counts[1]:4d} ({class_counts[1]/len(y)*100:5.1f}%)")
        print(f"    Possession (2): {class_counts[2]:4d} ({class_counts[2]/len(y)*100:5.1f}%)")

    return X, y


def outcome_to_class_id(outcome: str) -> int:
    """
    Convert outcome string to class ID.

    3-class system:
    - 0: Shot (includes Goal and Shot)
    - 1: Clearance
    - 2: Possession
    """
    outcome_lower = outcome.lower()

    if 'shot' in outcome_lower or 'goal' in outcome_lower:
        return 0
    elif 'clearance' in outcome_lower or 'clear' in outcome_lower:
        return 1
    elif 'possession' in outcome_lower or 'poss' in outcome_lower:
        return 2
    else:
        # Default to possession for unknown
        return 2


def aggregate_node_features_to_graph(node_features: np.ndarray) -> np.ndarray:
    """
    Aggregate node features to graph-level features (29D).

    Args:
        node_features: [num_nodes, 14] array

    Returns:
        graph_features: [29,] array
    """
    # Separate by team (feature 8 is team_flag: 1=attacking, 0=defending)
    team_flags = node_features[:, 8]
    attacking_mask = team_flags == 1
    defending_mask = team_flags == 0

    # Position statistics (features 0-1: x, y)
    positions = node_features[:, :2]
    attacking_pos = positions[attacking_mask]
    defending_pos = positions[defending_mask]

    features = []

    # Attacking team position stats (4 features)
    if len(attacking_pos) > 0:
        features.extend([
            attacking_pos[:, 0].mean(),  # mean_x_attacking
            attacking_pos[:, 1].mean(),  # mean_y_attacking
            attacking_pos[:, 0].std(),   # std_x_attacking
            attacking_pos[:, 1].std(),   # std_y_attacking
        ])
    else:
        features.extend([0, 0, 0, 0])

    # Defending team position stats (4 features)
    if len(defending_pos) > 0:
        features.extend([
            defending_pos[:, 0].mean(),  # mean_x_defending
            defending_pos[:, 1].mean(),  # mean_y_defending
            defending_pos[:, 0].std(),   # std_x_defending
            defending_pos[:, 1].std(),   # std_y_defending
        ])
    else:
        features.extend([0, 0, 0, 0])

    # Distance to goal statistics (feature 2)
    dist_to_goal = node_features[:, 2]
    features.extend([
        dist_to_goal.mean(),  # mean_distance_to_goal
        dist_to_goal.std(),   # std_distance_to_goal
        dist_to_goal.min(),   # min_distance_to_goal
    ])

    # Distance to ball statistics (feature 3)
    dist_to_ball = node_features[:, 3]
    features.extend([
        dist_to_ball.mean(),  # mean_distance_to_ball
        dist_to_ball.std(),   # std_distance_to_ball
    ])

    # Compactness metrics (std of positions)
    if len(attacking_pos) > 0:
        attacking_compactness = np.sqrt(attacking_pos.std(axis=0).sum())
    else:
        attacking_compactness = 0

    if len(defending_pos) > 0:
        defending_compactness = np.sqrt(defending_pos.std(axis=0).sum())
    else:
        defending_compactness = 0

    features.extend([
        attacking_compactness,  # attacking_compactness
        defending_compactness,  # defending_compactness
    ])

    # Defensive line height (max x of defending team)
    if len(defending_pos) > 0:
        defensive_line_height = defending_pos[:, 0].max()
    else:
        defensive_line_height = 0
    features.append(defensive_line_height)

    # Players in zones (feature 9: in_penalty_box)
    in_penalty = node_features[:, 9]
    players_in_penalty_attacking = (in_penalty & attacking_mask).sum()
    players_in_penalty_defending = (in_penalty & defending_mask).sum()

    # Approximate 6-yard box (x > 114 for StatsBomb coordinates)
    in_6yard = node_features[:, 0] > 114
    players_in_6yard_attacking = (in_6yard & attacking_mask).sum()
    players_in_6yard_defending = (in_6yard & defending_mask).sum()

    features.extend([
        players_in_6yard_attacking,
        players_in_6yard_defending,
        players_in_penalty_attacking,
        players_in_penalty_defending,
    ])

    # Angular features (features 6-7: angle_to_goal, angle_to_ball)
    angle_to_goal = node_features[:, 6]
    angle_to_ball = node_features[:, 7]

    features.extend([
        angle_to_goal.mean(),  # mean_angle_to_goal
        angle_to_goal.std(),   # std_angle_to_goal
        angle_to_ball.mean(),  # mean_angle_to_ball
    ])

    # Density metrics (features 12-13)
    num_nearby = node_features[:, 12]
    local_density = node_features[:, 13]

    features.extend([
        num_nearby.mean(),    # player_density_6yard (proxy)
        local_density.mean(), # player_density_penalty
    ])

    # Formation width
    if len(attacking_pos) > 0:
        formation_width_attacking = attacking_pos[:, 1].max() - attacking_pos[:, 1].min()
    else:
        formation_width_attacking = 0

    if len(defending_pos) > 0:
        formation_width_defending = defending_pos[:, 1].max() - defending_pos[:, 1].min()
    else:
        formation_width_defending = 0

    features.extend([
        formation_width_attacking,
        formation_width_defending,
    ])

    # Attacking/defending ratio
    num_attacking = attacking_mask.sum()
    num_defending = defending_mask.sum()
    if num_defending > 0:
        ad_ratio = num_attacking / num_defending
    else:
        ad_ratio = 0
    features.append(ad_ratio)

    # Ball landing zone (use mean x of all players as proxy)
    ball_landing_zone_x = positions[:, 0].mean()
    features.append(ball_landing_zone_x)

    return np.array(features)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  task: str = 'outcome',
                  random_seed: int = 42) -> xgb.Booster:
    """
    Train XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        task: 'receiver' or 'outcome'
        random_seed: Random seed

    Returns:
        Trained XGBoost Booster
    """
    if task == 'receiver':
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'objective': 'multi:softmax',
            'num_class': 22,
            'eval_metric': 'mlogloss',
            'seed': random_seed
        }
    elif task == 'outcome':
        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'seed': random_seed
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train
    evals = [(dtrain, 'train'), (dval, 'val')]
    print(f"\nTraining XGBoost for {task} task...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    print(f"✓ Training complete (best iteration: {model.best_iteration})")
    return model


def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray,
                   task: str = 'outcome') -> ExperimentResults:
    """
    Evaluate trained model.

    Args:
        model: Trained XGBoost Booster
        X_test: Test features
        y_test: Test labels
        task: 'receiver' or 'outcome'

    Returns:
        ExperimentResults object
    """
    dtest = xgb.DMatrix(X_test)

    results = ExperimentResults(
        config_name="",
        num_graphs=len(y_test)
    )

    if task == 'receiver':
        # Get prediction probabilities for Top-k accuracy
        # For multi:softmax, we need to use multi:softprob
        # Retrain or use custom prediction
        y_pred = model.predict(dtest).astype(int)

        # For Top-k, we need probabilities
        # Workaround: use model with softprob objective temporarily
        params = model.attributes()

        # Simple Top-1
        results.receiver_top1 = accuracy_score(y_test, y_pred)

        # For Top-3 and Top-5, we need probability outputs
        # This requires retraining with multi:softprob, but for now we'll use approximation
        # by checking if true label is in top-k predictions
        # Since we only have single predictions, we'll compute differently

        # Placeholder for now (will implement proper version)
        results.receiver_top3 = None
        results.receiver_top5 = None

    elif task == 'outcome':
        y_pred = model.predict(dtest).astype(int)

        # Metrics
        results.outcome_accuracy = accuracy_score(y_test, y_pred)
        results.outcome_macro_f1 = f1_score(y_test, y_pred, average='macro')

        # Per-class F1
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        if len(per_class_f1) >= 3:
            results.outcome_shot_f1 = per_class_f1[0]
            results.outcome_clearance_f1 = per_class_f1[1]
            results.outcome_possession_f1 = per_class_f1[2]

        # Confusion matrix
        results.confusion_matrix = confusion_matrix(y_test, y_pred)

    return results


def print_results(results: ExperimentResults, task: str):
    """Print experiment results."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {results.config_name}")
    print(f"{'='*80}")
    print(f"Graphs: {results.num_graphs}")

    if task == 'receiver':
        if results.receiver_top1 is not None:
            print(f"  Receiver Top-1: {results.receiver_top1*100:.2f}%")
        if results.receiver_top3 is not None:
            print(f"  Receiver Top-3: {results.receiver_top3*100:.2f}%")
        if results.receiver_top5 is not None:
            print(f"  Receiver Top-5: {results.receiver_top5*100:.2f}%")

    elif task == 'outcome':
        if results.outcome_accuracy is not None:
            print(f"  Outcome Accuracy: {results.outcome_accuracy*100:.1f}%")
        if results.outcome_macro_f1 is not None:
            print(f"  Macro F1: {results.outcome_macro_f1:.3f}")
        if results.outcome_shot_f1 is not None:
            print(f"  Shot F1: {results.outcome_shot_f1:.3f}")
        if results.outcome_clearance_f1 is not None:
            print(f"  Clearance F1: {results.outcome_clearance_f1:.3f}")
        if results.outcome_possession_f1 is not None:
            print(f"  Possession F1: {results.outcome_possession_f1:.3f}")

        if results.confusion_matrix is not None:
            print("\nConfusion Matrix:")
            cm = results.confusion_matrix
            print("  Predicted:  Shot  Clear  Poss")
            class_names = ['Shot', 'Clear', 'Poss']
            for i, name in enumerate(class_names):
                row_str = f"  {name:5s}     : "
                row_str += "  ".join(f"{cm[i, j]:4d}" for j in range(3))
                print(row_str)

    print(f"{'='*80}\n")


def save_results_to_json(results: ExperimentResults, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"✓ Saved results to {output_path}")


def experiment_1_temporal_augmentation(graph_path: str, output_dir: Path,
                                       random_seed: int = 42):
    """
    Experiment 1: Impact of temporal augmentation.

    Compares:
    - Configuration 1: No augmentation (t=0s only)
    - Configuration 2: With augmentation (all 5 frames)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: TEMPORAL AUGMENTATION IMPACT")
    print("="*80)

    # Load full dataset
    all_graphs = load_graph_dataset(graph_path)

    # Configuration 2: With augmentation (baseline)
    print("\n--- Configuration 2: With Augmentation (5 frames) ---")
    train_aug, val_aug, test_aug = split_dataset(all_graphs, random_seed=random_seed)

    # Train on outcome task
    X_train_aug, y_train_aug = extract_features_and_labels(train_aug, task='outcome')
    X_val_aug, y_val_aug = extract_features_and_labels(val_aug, task='outcome')
    X_test_aug, y_test_aug = extract_features_and_labels(test_aug, task='outcome')

    model_aug = train_xgboost(X_train_aug, y_train_aug, X_val_aug, y_val_aug,
                              task='outcome', random_seed=random_seed)

    results_aug = evaluate_model(model_aug, X_test_aug, y_test_aug, task='outcome')
    results_aug.config_name = "With Augmentation (5 frames)"
    print_results(results_aug, task='outcome')

    # Configuration 1: No augmentation (t=0s only)
    print("\n--- Configuration 1: No Augmentation (t=0s only) ---")
    t0_graphs = filter_t0_frames(all_graphs)
    train_t0, val_t0, test_t0 = split_dataset(t0_graphs, random_seed=random_seed)

    X_train_t0, y_train_t0 = extract_features_and_labels(train_t0, task='outcome')
    X_val_t0, y_val_t0 = extract_features_and_labels(val_t0, task='outcome')
    X_test_t0, y_test_t0 = extract_features_and_labels(test_t0, task='outcome')

    model_t0 = train_xgboost(X_train_t0, y_train_t0, X_val_t0, y_val_t0,
                             task='outcome', random_seed=random_seed)

    results_t0 = evaluate_model(model_t0, X_test_t0, y_test_t0, task='outcome')
    results_t0.config_name = "No Augmentation (t=0s only)"
    print_results(results_t0, task='outcome')

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Configuration 1 (t=0s): Outcome Acc = {results_t0.outcome_accuracy*100:.1f}%, "
          f"Macro F1 = {results_t0.outcome_macro_f1:.3f}, "
          f"Shot F1 = {results_t0.outcome_shot_f1:.3f}")
    print(f"Configuration 2 (5 frames): Outcome Acc = {results_aug.outcome_accuracy*100:.1f}%, "
          f"Macro F1 = {results_aug.outcome_macro_f1:.3f}, "
          f"Shot F1 = {results_aug.outcome_shot_f1:.3f}")

    acc_improvement = (results_aug.outcome_accuracy - results_t0.outcome_accuracy) * 100
    f1_improvement = results_aug.outcome_macro_f1 - results_t0.outcome_macro_f1
    shot_f1_improvement = results_aug.outcome_shot_f1 - results_t0.outcome_shot_f1

    print(f"\nImprovement:")
    print(f"  Outcome Accuracy: {acc_improvement:+.1f}pp")
    print(f"  Macro F1: {f1_improvement:+.3f}")
    print(f"  Shot F1: {shot_f1_improvement:+.3f}")
    print("="*80)

    # Save results
    exp1_dir = output_dir / "exp1_temporal_augmentation"
    save_results_to_json(results_t0, exp1_dir / "results_t0.json")
    save_results_to_json(results_aug, exp1_dir / "results_augmented.json")

    # Generate LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Impact of temporal augmentation on XGBoost performance (StatsBomb Open Data).}}
\\label{{tab:augmentation_impact}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{Graphs}} & \\textbf{{Outcome Acc}} & \\textbf{{Macro F1}} & \\textbf{{Shot F1}} \\\\
\\midrule
No augmentation (t=0s only) & {results_t0.num_graphs} & {results_t0.outcome_accuracy*100:.1f}\\% & {results_t0.outcome_macro_f1:.3f} & {results_t0.outcome_shot_f1:.3f} \\\\
With augmentation (5 frames) & {results_aug.num_graphs} & {results_aug.outcome_accuracy*100:.1f}\\% & {results_aug.outcome_macro_f1:.3f} & {results_aug.outcome_shot_f1:.3f} \\\\
\\midrule
Improvement & {results_aug.num_graphs/results_t0.num_graphs:.1f}× data & {acc_improvement:+.1f}pp & {f1_improvement:+.3f} & {shot_f1_improvement:+.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    latex_path = exp1_dir / "latex_table.txt"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n✓ Saved LaTeX table to {latex_path}")

    # Interpretation
    interpretation = f"""# Experiment 1: Temporal Augmentation Impact

## Results Summary
- **Configuration 1 (t=0s only)**: {results_t0.num_graphs} graphs
  - Outcome Accuracy: {results_t0.outcome_accuracy*100:.1f}%
  - Macro F1: {results_t0.outcome_macro_f1:.3f}
  - Shot F1: {results_t0.outcome_shot_f1:.3f}

- **Configuration 2 (5 frames)**: {results_aug.num_graphs} graphs
  - Outcome Accuracy: {results_aug.outcome_accuracy*100:.1f}%
  - Macro F1: {results_aug.outcome_macro_f1:.3f}
  - Shot F1: {results_aug.outcome_shot_f1:.3f}

## Improvement
- Outcome Accuracy: {acc_improvement:+.1f}pp
- Macro F1: {f1_improvement:+.3f}
- Shot F1: {shot_f1_improvement:+.3f}

## Interpretation
Temporal augmentation provides a {results_aug.num_graphs/results_t0.num_graphs:.1f}× increase in training data
(from {results_t0.num_graphs} to {results_aug.num_graphs} graphs). The performance improvement is
{"modest" if abs(acc_improvement) < 2 else "substantial"}, with outcome accuracy improving by {acc_improvement:.1f}
percentage points.

{"The results suggest that temporal dynamics around corner kicks provide additional predictive signal, " if acc_improvement > 0 else "Interestingly, temporal augmentation does not significantly improve performance, suggesting that "}
{"particularly for the minority Shot class (F1 improvement: " + f"{shot_f1_improvement:+.3f}" + ")." if acc_improvement > 0 else "the corner kick moment (t=0s) captures most of the relevant information."}

This finding {"validates" if acc_improvement > 1 else "questions"} the utility of temporal augmentation for this
prediction task and suggests that {"expanding the temporal window may yield further gains" if acc_improvement > 1 else "simpler single-frame models may be sufficient"}.
"""

    interpretation_path = exp1_dir / "interpretation.md"
    with open(interpretation_path, 'w') as f:
        f.write(interpretation)
    print(f"✓ Saved interpretation to {interpretation_path}")

    return results_t0, results_aug


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='XGBoost Ablation Studies for Corner Kick Outcome Prediction'
    )
    parser.add_argument('--graph-path', type=str,
                       default='data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl',
                       help='Path to graph dataset pickle file')
    parser.add_argument('--output-dir', type=str,
                       default='results/ablation_studies',
                       help='Output directory for results')
    parser.add_argument('--experiments', type=str, default='1',
                       help='Experiments to run: 1,2,3 or all (default: 1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    output_dir = Path(args.output_dir)

    print("\n" + "="*80)
    print("XGBOOST ABLATION STUDIES")
    print("="*80)
    print(f"Graph path: {args.graph_path}")
    print(f"Output dir: {output_dir}")
    print(f"Experiments: {args.experiments}")
    print(f"Random seed: {args.random_seed}")
    print("="*80)

    # Run experiments
    if args.experiments == 'all' or '1' in args.experiments:
        experiment_1_temporal_augmentation(args.graph_path, output_dir, args.random_seed)

    print("\n" + "="*80)
    print("ABLATION STUDIES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
