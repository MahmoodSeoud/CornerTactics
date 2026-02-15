"""Baseline 3: XGBoost on aggregate features (shot prediction only).

Extracts hand-crafted aggregate features from each 22-player graph
and trains an XGBoost classifier for shot prediction.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.evaluate import (
    compute_receiver_metrics,
    compute_shot_metrics,
)

logger = logging.getLogger(__name__)

# Goal center in normalized coordinates
GOAL_X = 1.0
GOAL_Y = 0.0

# Penalty box bounds in normalized coordinates (approx 16.5m from goal line, 20.15m each side)
BOX_X_MIN = (52.5 - 16.5) / 52.5   # ~0.686
BOX_Y_ABS_MAX = 20.15 / 34.0        # ~0.593


def extract_features(graph) -> np.ndarray:
    """Extract aggregate feature vector from a corner kick graph.

    Returns a 1D numpy array of 27 features.
    """
    x = graph.x.numpy()  # [22, 13]
    n_nodes = x.shape[0]

    # Split by team
    is_atk = x[:, 5].astype(bool)
    is_def = ~is_atk

    atk_x, atk_y = x[is_atk, 0], x[is_atk, 1]
    def_x, def_y = x[is_def, 0], x[is_def, 1]

    atk_vx, atk_vy, atk_speed = x[is_atk, 2], x[is_atk, 3], x[is_atk, 4]
    def_vx, def_vy, def_speed = x[is_def, 2], x[is_def, 3], x[is_def, 4]

    # Distance to goal
    def _dist_to_goal(px, py):
        return np.sqrt((px - GOAL_X) ** 2 + (py - GOAL_Y) ** 2)

    atk_dist_goal = _dist_to_goal(atk_x, atk_y)
    def_dist_goal = _dist_to_goal(def_x, def_y)

    # Players in penalty box
    def _in_box(px, py):
        return ((px >= BOX_X_MIN) & (np.abs(py) <= BOX_Y_ABS_MAX)).sum()

    features = []

    # Attacker spatial (8)
    features.extend([
        _safe_mean(atk_x), _safe_mean(atk_y),
        _safe_std(atk_x), _safe_std(atk_y),
        _safe_mean(atk_dist_goal), _safe_min(atk_dist_goal),
        float(_in_box(atk_x, atk_y)),
        float(is_atk.sum()),
    ])

    # Defender spatial (8)
    features.extend([
        _safe_mean(def_x), _safe_mean(def_y),
        _safe_std(def_x), _safe_std(def_y),
        _safe_mean(def_dist_goal), _safe_min(def_dist_goal),
        float(_in_box(def_x, def_y)),
        float(is_def.sum()),
    ])

    # Attacker velocity (4)
    features.extend([
        _safe_mean(atk_vx), _safe_mean(atk_vy),
        _safe_mean(atk_speed), _safe_max(atk_speed),
    ])

    # Defender velocity (4)
    features.extend([
        _safe_mean(def_vx), _safe_mean(def_vy),
        _safe_mean(def_speed), _safe_max(def_speed),
    ])

    # Speed differential (1)
    features.append(_safe_mean(atk_speed) - _safe_mean(def_speed))

    # Graph-level (2)
    features.append(float(graph.corner_side))
    features.append(float(graph.detection_rate))

    return np.array(features, dtype=np.float64)


def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_std(arr):
    return float(np.std(arr)) if len(arr) > 0 else 0.0


def _safe_min(arr):
    return float(np.min(arr)) if len(arr) > 0 else 0.0


def _safe_max(arr):
    return float(np.max(arr)) if len(arr) > 0 else 0.0


FEATURE_NAMES = [
    "mean_atk_x", "mean_atk_y", "std_atk_x", "std_atk_y",
    "mean_atk_dist_goal", "min_atk_dist_goal", "n_atk_in_box", "n_attackers",
    "mean_def_x", "mean_def_y", "std_def_x", "std_def_y",
    "mean_def_dist_goal", "min_def_dist_goal", "n_def_in_box", "n_defenders",
    "mean_atk_vx", "mean_atk_vy", "mean_atk_speed", "max_atk_speed",
    "mean_def_vx", "mean_def_vy", "mean_def_speed", "max_def_speed",
    "speed_diff", "corner_side", "detection_rate",
]


def _xgboost_fold(train_data: List, test_data: List, seed: int) -> Dict:
    """Train and evaluate XGBoost on one fold."""
    X_train = np.array([extract_features(g) for g in train_data])
    y_train = np.array([g.shot_label for g in train_data])
    X_test = np.array([extract_features(g) for g in test_data])
    y_test = np.array([g.shot_label for g in test_data], dtype=float)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    n_samples = len(y_test)

    # AUC
    if len(np.unique(y_test)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_test, probs)

    # F1 at optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_05 = (probs >= 0.5).astype(int)
    accuracy = float((preds_05 == y_test).mean())

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "accuracy": accuracy,
        "probs": probs.tolist(),
        "labels": y_test.tolist(),
        "n_samples": n_samples,
        "n_positive": int(y_test.sum()),
    }


def xgboost_baseline_lomo(
    dataset, seed: int = 42, verbose: bool = True,
) -> Dict[str, Any]:
    """Run XGBoost baseline with LOMO cross-validation.

    Shot prediction only — no receiver prediction.
    """
    match_ids = get_match_ids(dataset)
    fold_results = []

    dummy_receiver = {
        "top1_acc": 0.0, "top3_acc": 0.0, "n_labeled": 0, "per_graph": [],
    }

    for fold_idx, held_out in enumerate(match_ids):
        if verbose:
            print(f"\n--- XGBoost Baseline Fold {fold_idx + 1}/{len(match_ids)}: "
                  f"held_out={held_out} ---")

        train_data, test_data = lomo_split(dataset, held_out)
        if not test_data:
            continue

        shot = _xgboost_fold(train_data, test_data, seed)

        fold_result = {
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(train_data),
            "n_val": 0,
            "n_test": len(test_data),
            "receiver": dummy_receiver,
            "shot_oracle": shot,
            "shot_predicted": shot,
            "shot_unconditional": shot,
        }
        fold_results.append(fold_result)

        if verbose:
            n_train_shots = sum(1 for g in train_data if g.shot_label == 1)
            print(f"  train={len(train_data)} ({n_train_shots} shots), "
                  f"test={len(test_data)}")
            print(f"  Shot: AUC={shot['auc']:.3f}, F1={shot['f1']:.3f}")

    agg_shot = compute_shot_metrics(fold_results, "oracle")

    dummy_agg_receiver = {
        "top1_mean": 0.0, "top1_std": 0.0, "top3_mean": 0.0, "top3_std": 0.0,
        "n_folds": 0, "per_fold_top1": [], "per_fold_top3": [],
    }

    results = {
        "config": {
            "baseline": "xgboost",
            "seed": seed,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "n_folds": len(fold_results),
        },
        "per_fold": fold_results,
        "aggregated": {
            "receiver": dummy_agg_receiver,
            "shot_oracle": agg_shot,
            "shot_predicted": agg_shot,
            "shot_unconditional": agg_shot,
        },
    }

    if verbose:
        _print_xgboost_results(results)

    return results


def _print_xgboost_results(results: Dict) -> None:
    s = results["aggregated"]["shot_oracle"]
    print(f"\n{'=' * 60}")
    print("XGBoost Baseline Results (Shot Prediction Only)")
    print(f"{'=' * 60}")
    print(f"Features: {results['config']['n_features']} aggregate features")
    print(f"Shot AUC: {s['auc_mean']:.3f} +/- {s['auc_std']:.3f}")
    print(f"Shot F1:  {s['f1_mean']:.3f} +/- {s['f1_std']:.3f}")
    print(f"{'=' * 60}")
