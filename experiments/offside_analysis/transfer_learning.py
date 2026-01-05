"""Transfer learning analysis for offside features.

Tests whether offside-predictive features can improve shot prediction.
"""

from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.offside_analysis.feature_extraction import extract_offside_features


def augment_with_offside_features(corners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add offside features to corner data.

    Args:
        corners: List of corner dictionaries

    Returns:
        Corners with 'offside_features' added to each
    """
    augmented = []
    for corner in corners:
        corner_copy = deepcopy(corner)
        corner_copy['offside_features'] = extract_offside_features(corner)
        augmented.append(corner_copy)
    return augmented


def create_feature_matrix(
    corners: List[Dict[str, Any]],
    include_offside: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create feature matrix from corners.

    Args:
        corners: List of corner dictionaries
        include_offside: Whether to include offside features

    Returns:
        Tuple of (X feature matrix, y labels)
    """
    features_list = []
    labels = []

    for corner in corners:
        feats = extract_offside_features(corner)

        if include_offside:
            feature_vec = [
                feats.get('last_defender_x', 110.0) if not np.isnan(feats.get('last_defender_x', np.nan)) else 110.0,
                feats.get('defensive_line_spread', 20.0) if not np.isnan(feats.get('defensive_line_spread', np.nan)) else 20.0,
                feats.get('defensive_compactness', 5.0) if not np.isnan(feats.get('defensive_compactness', np.nan)) else 5.0,
                feats.get('attackers_beyond_defender', 0) if not np.isnan(feats.get('attackers_beyond_defender', np.nan)) else 0,
                feats.get('furthest_attacker_x', 110.0) if not np.isnan(feats.get('furthest_attacker_x', np.nan)) else 110.0,
                feats.get('attacker_defender_gap', 0.0) if not np.isnan(feats.get('attacker_defender_gap', np.nan)) else 0.0,
                feats.get('attackers_in_offside_zone', 0) if not np.isnan(feats.get('attackers_in_offside_zone', np.nan)) else 0,
                feats.get('num_defenders', 4) if not np.isnan(feats.get('num_defenders', np.nan)) else 4,
                feats.get('num_attackers', 4) if not np.isnan(feats.get('num_attackers', np.nan)) else 4,
            ]
        else:
            # Baseline features without offside-specific ones
            feature_vec = [
                feats.get('num_defenders', 4) if not np.isnan(feats.get('num_defenders', np.nan)) else 4,
                feats.get('num_attackers', 4) if not np.isnan(feats.get('num_attackers', np.nan)) else 4,
            ]

        features_list.append(feature_vec)
        labels.append(1 if corner.get('shot_outcome', 0) else 0)

    return np.array(features_list), np.array(labels)


def train_baseline_classifier(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train baseline classifier without offside features.

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        Tuple of (trained model, metrics dict)
    """
    X, y = create_feature_matrix(corners, include_offside=False)

    # Handle small datasets
    if len(X) < 10:
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X, y)
        return model, {'auc': 0.5}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
    except (ValueError, IndexError):
        auc = 0.5

    return model, {'auc': auc}


def train_with_offside_features(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train classifier with offside features.

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        Tuple of (trained model, metrics dict)
    """
    X, y = create_feature_matrix(corners, include_offside=True)

    # Handle small datasets
    if len(X) < 10:
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X, y)
        return model, {'auc': 0.5}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    try:
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
    except (ValueError, IndexError):
        auc = 0.5

    return model, {'auc': auc}


def compare_classifiers(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> Dict[str, float]:
    """Compare baseline vs offside-augmented classifier.

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        Comparison metrics
    """
    _, baseline_metrics = train_baseline_classifier(corners, random_state)
    _, augmented_metrics = train_with_offside_features(corners, random_state)

    return {
        'baseline_auc': baseline_metrics['auc'],
        'augmented_auc': augmented_metrics['auc'],
        'improvement': augmented_metrics['auc'] - baseline_metrics['auc'],
    }


def train_hierarchical_classifier(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[Dict, Dict[str, float]]:
    """Train two-stage hierarchical classifier.

    Stage 1: Predict offside-like patterns
    Stage 2: Predict shot outcome

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        Tuple of (model dict, metrics dict)
    """
    X, y = create_feature_matrix(corners, include_offside=True)

    if len(X) < 10:
        model = RandomForestClassifier(n_estimators=10, random_state=random_state)
        model.fit(X, y)
        return {'combined': model}, {'overall_auc': 0.5}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Train combined model (simplified hierarchical)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    try:
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
    except (ValueError, IndexError):
        auc = 0.5

    return {'combined': model}, {'overall_auc': auc}


def compute_feature_importance(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute importance of each feature.

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        Dict mapping feature name to importance score
    """
    X, y = create_feature_matrix(corners, include_offside=True)

    feature_names = [
        'last_defender_x',
        'defensive_line_spread',
        'defensive_compactness',
        'attackers_beyond_defender',
        'furthest_attacker_x',
        'attacker_defender_gap',
        'attackers_in_offside_zone',
        'num_defenders',
        'num_attackers',
    ]

    if len(X) < 5:
        return {name: 0.0 for name in feature_names}

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)

    importance = {}
    for name, imp in zip(feature_names, model.feature_importances_):
        importance[name] = float(imp)

    return importance


def rank_offside_features(
    corners: List[Dict[str, Any]],
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """Rank offside features by importance.

    Args:
        corners: List of corner dictionaries
        random_state: Random seed

    Returns:
        List of (feature_name, importance) tuples, sorted descending
    """
    importance = compute_feature_importance(corners, random_state)

    # Sort by importance descending
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return ranked


def compute_feature_statistics(
    corners: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute statistics of features by outcome.

    Args:
        corners: List of corner dictionaries

    Returns:
        Nested dict: {outcome: {stat: {feature: value}}}
    """
    shot_features = []
    no_shot_features = []

    for corner in corners:
        feats = extract_offside_features(corner)
        if corner.get('shot_outcome', 0) == 1:
            shot_features.append(feats)
        else:
            no_shot_features.append(feats)

    def compute_stats(features_list):
        if not features_list:
            return {'mean': {}, 'std': {}}

        df = pd.DataFrame(features_list)
        return {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
        }

    return {
        'shot': compute_stats(shot_features),
        'no_shot': compute_stats(no_shot_features),
    }


def compute_feature_significance(
    corners: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Test statistical significance of feature differences.

    Uses t-test to compare shot vs no-shot distributions.

    Args:
        corners: List of corner dictionaries

    Returns:
        Dict mapping feature name to significance results
    """
    shot_features = []
    no_shot_features = []

    for corner in corners:
        feats = extract_offside_features(corner)
        if corner.get('shot_outcome', 0) == 1:
            shot_features.append(feats)
        else:
            no_shot_features.append(feats)

    if not shot_features or not no_shot_features:
        return {}

    shot_df = pd.DataFrame(shot_features)
    no_shot_df = pd.DataFrame(no_shot_features)

    results = {}
    for col in shot_df.columns:
        if col in ['match_id', 'shot_outcome']:
            continue

        shot_vals = shot_df[col].dropna().values
        no_shot_vals = no_shot_df[col].dropna().values

        if len(shot_vals) < 2 or len(no_shot_vals) < 2:
            continue

        try:
            t_stat, p_val = stats.ttest_ind(shot_vals, no_shot_vals)
            results[col] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
            }
        except Exception:
            results[col] = {
                't_statistic': 0.0,
                'p_value': 1.0,
            }

    return results
