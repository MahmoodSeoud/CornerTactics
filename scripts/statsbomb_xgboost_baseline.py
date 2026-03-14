#!/usr/bin/env python3
"""XGBoost position-only baseline on StatsBomb freeze-frames and tracking data.

Runs XGBoost with 18 position-only aggregate features (dropping 9 velocity-
dependent features from the full 27) on:
  1. StatsBomb freeze-frame data (grouped k-fold CV)
  2. Tracking data — combined 143 corners (LOMO CV)

StatsBomb velocity is already zero (single frame), but we explicitly drop the
velocity features to match the experiment design (18 features, not 27 with 9
zeros). This gives a clean three-way comparison:

  | Condition             | Data        | Features         |
  |-----------------------|-------------|------------------|
  | Tracking + velocity   | 143 SK+DFL  | 27 aggregate     |
  | Tracking pos-only     | 143 SK+DFL  | 18 position-only |
  | Freeze-frame pos-only | N SB        | 18 position-only |

Usage:
    cd /home/mseo/CornerTactics
    source FAANTRA/venv/bin/activate

    # StatsBomb only (default)
    python scripts/statsbomb_xgboost_baseline.py

    # Tracking position-only
    python scripts/statsbomb_xgboost_baseline.py --tracking-posonly

    # Everything with permutation tests
    python scripts/statsbomb_xgboost_baseline.py --all

    # Just permutation tests
    python scripts/statsbomb_xgboost_baseline.py --permutation
"""

import argparse
import copy
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corner_prediction.baselines.xgboost_baseline import (
    FEATURE_NAMES,
    extract_features,
)
from corner_prediction.config import RESULTS_DIR
from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.permutation_test import shuffle_shot_labels

# ---- Constants ----------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1234]

# Position-only feature indices from the 27-feature vector:
#   [0:16]  = attacker spatial (8) + defender spatial (8)
#   [25]    = corner_side
#   [26]    = detection_rate
# Dropped: [16:25] = attacker velocity (4) + defender velocity (4) + speed_diff (1)
POSITION_INDICES = list(range(16)) + [25, 26]

POSITION_FEATURE_NAMES = [
    "mean_atk_x", "mean_atk_y", "std_atk_x", "std_atk_y",
    "mean_atk_dist_goal", "min_atk_dist_goal", "n_atk_in_box", "n_attackers",
    "mean_def_x", "mean_def_y", "std_def_x", "std_def_y",
    "mean_def_dist_goal", "min_def_dist_goal", "n_def_in_box", "n_defenders",
    "corner_side", "detection_rate",
]


# ---- Feature extraction ------------------------------------------------------

def extract_position_features(graph) -> np.ndarray:
    """Extract 18 position-only aggregate features from a graph.

    Calls extract_features() for the full 27 features, then keeps only
    the 16 spatial + 2 graph-level features (dropping 9 velocity-dependent).
    """
    full = extract_features(graph)
    return full[POSITION_INDICES]


# ---- XGBoost fold -------------------------------------------------------------

def _xgboost_posonly_fold(
    train_data: List,
    test_data: List,
    seed: int,
) -> Dict[str, Any]:
    """Train and evaluate XGBoost with position-only features on one fold."""
    X_train = np.array([extract_position_features(g) for g in train_data])
    y_train = np.array([g.shot_label for g in train_data])
    X_test = np.array([extract_position_features(g) for g in test_data])
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
        "n_samples": len(y_test),
        "n_positive": int(y_test.sum()),
    }


# ---- Grouped k-fold CV (StatsBomb) -------------------------------------------

def grouped_kfold_xgboost(
    dataset: List,
    n_folds: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Grouped k-fold CV for XGBoost position-only on StatsBomb data.

    Groups by match_id so no match leaks across train/test.
    XGBoost needs no validation set (no early stopping), so all non-test
    data is used for training.
    """
    match_ids = sorted(set(str(g.match_id) for g in dataset))
    rng = np.random.RandomState(seed)
    shuffled_ids = list(match_ids)
    rng.shuffle(shuffled_ids)
    fold_matches = [shuffled_ids[i::n_folds] for i in range(n_folds)]

    fold_aucs = []
    fold_f1s = []
    fold_details = []

    for fold_idx in range(n_folds):
        test_match_set = set(fold_matches[fold_idx])
        test_data = [g for g in dataset if str(g.match_id) in test_match_set]
        train_data = [g for g in dataset if str(g.match_id) not in test_match_set]

        if not test_data:
            continue

        test_labels = [g.shot_label for g in test_data]
        if len(set(test_labels)) < 2:
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_folds}: skipped (single class in test)")
            continue

        result = _xgboost_posonly_fold(train_data, test_data, seed)
        fold_aucs.append(result["auc"])
        fold_f1s.append(result["f1"])

        fold_details.append({
            "fold_idx": fold_idx,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "test_matches": sorted(test_match_set),
            **result,
        })

        if verbose:
            n_train_shots = sum(1 for g in train_data if g.shot_label == 1)
            print(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_data)} "
                  f"({n_train_shots} shots), test={len(test_data)}, "
                  f"AUC={result['auc']:.3f}, F1={result['f1']:.3f}")

    auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    auc_std = float(np.std(fold_aucs)) if fold_aucs else 0.0
    f1_mean = float(np.mean(fold_f1s)) if fold_f1s else 0.0
    f1_std = float(np.std(fold_f1s)) if fold_f1s else 0.0

    return {
        "config": {
            "baseline": "xgboost_position_only",
            "seed": seed,
            "n_folds": n_folds,
            "n_features": len(POSITION_FEATURE_NAMES),
            "feature_names": POSITION_FEATURE_NAMES,
            "cv_method": f"grouped_{n_folds}fold",
        },
        "per_fold": fold_details,
        "aggregated": {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "n_valid_folds": len(fold_aucs),
            "per_fold_aucs": fold_aucs,
            "per_fold_f1s": fold_f1s,
        },
        "n_corners": len(dataset),
        "n_matches": len(match_ids),
    }


# ---- LOMO CV (Tracking data) -------------------------------------------------

def lomo_xgboost_posonly(
    dataset: List,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """LOMO CV for XGBoost position-only on tracking data (143 corners)."""
    match_ids = get_match_ids(dataset)
    fold_aucs = []
    fold_f1s = []
    fold_details = []

    for fold_idx, held_out in enumerate(match_ids):
        train_data, test_data = lomo_split(dataset, held_out)
        if not test_data:
            continue

        test_labels = [g.shot_label for g in test_data]
        if len(set(test_labels)) < 2:
            if verbose:
                print(f"  Fold {fold_idx+1}/{len(match_ids)}: held_out={held_out} "
                      f"skipped (single class)")
            continue

        result = _xgboost_posonly_fold(train_data, test_data, seed)
        fold_aucs.append(result["auc"])
        fold_f1s.append(result["f1"])

        fold_details.append({
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(train_data),
            "n_test": len(test_data),
            **result,
        })

        if verbose:
            n_train_shots = sum(1 for g in train_data if g.shot_label == 1)
            print(f"  Fold {fold_idx+1}/{len(match_ids)}: held_out={held_out}, "
                  f"train={len(train_data)} ({n_train_shots} shots), "
                  f"test={len(test_data)}, AUC={result['auc']:.3f}, "
                  f"F1={result['f1']:.3f}")

    auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    auc_std = float(np.std(fold_aucs)) if fold_aucs else 0.0
    f1_mean = float(np.mean(fold_f1s)) if fold_f1s else 0.0
    f1_std = float(np.std(fold_f1s)) if fold_f1s else 0.0

    return {
        "config": {
            "baseline": "xgboost_position_only",
            "seed": seed,
            "n_folds": len(match_ids),
            "n_features": len(POSITION_FEATURE_NAMES),
            "feature_names": POSITION_FEATURE_NAMES,
            "cv_method": "lomo",
        },
        "per_fold": fold_details,
        "aggregated": {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "n_valid_folds": len(fold_aucs),
            "per_fold_aucs": fold_aucs,
            "per_fold_f1s": fold_f1s,
        },
        "n_corners": len(dataset),
        "n_matches": len(match_ids),
    }


# ---- Permutation tests -------------------------------------------------------

def permutation_test_xgboost(
    dataset: List,
    cv_func,
    n_permutations: int = 100,
    seed: int = 42,
    verbose: bool = True,
    **cv_kwargs,
) -> Dict[str, Any]:
    """Permutation test for XGBoost position-only shot prediction.

    Args:
        dataset: List of PyG Data objects.
        cv_func: CV function (grouped_kfold_xgboost or lomo_xgboost_posonly).
        n_permutations: Number of permutations.
        seed: Random seed.
        **cv_kwargs: Extra kwargs passed to cv_func (e.g., n_folds).
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: XGBoost Position-Only (n={n_permutations})")
        print(f"{'=' * 60}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_results = cv_func(dataset, seed=seed, verbose=verbose, **cv_kwargs)
    real_auc = real_results["aggregated"]["auc_mean"]

    if verbose:
        print(f"\nReal AUC: {real_auc:.4f}")

    # Null distribution
    null_aucs = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_results = cv_func(shuffled, seed=seed, verbose=False, **cv_kwargs)
        null_aucs.append(perm_results["aggregated"]["auc_mean"])

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= real_auc) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_aucs.mean():.4f}, "
              f"std={null_aucs.std():.4f}")
        print(f"p-value: {p_value:.4f} "
              f"{'***' if p_value < 0.01 else '**' if p_value < 0.05 else '(not significant)'}")

    return {
        "metric": "shot_auc",
        "real_metric": float(real_auc),
        "null_distribution": null_aucs.tolist(),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": bool(p_value < 0.05),
        "cv_method": real_results["config"]["cv_method"],
    }


# ---- Data loading -------------------------------------------------------------

def load_statsbomb_dataset() -> List:
    """Load StatsBomb freeze-frame data as USSF-aligned PyG graphs."""
    from scripts.statsbomb_ussf_pipeline import build_dataset
    return build_dataset()


def load_tracking_dataset() -> List:
    """Load combined tracking data (143 corners) as USSF-aligned PyG graphs."""
    from corner_prediction.data.dataset import CornerKickDataset

    data_dir = Path(__file__).resolve().parent.parent / "corner_prediction" / "data"
    ds = CornerKickDataset(
        root=str(data_dir),
        records_file="combined_corners.pkl",
        edge_type="dense",
        k=6,
        feature_mode="ussf_aligned",
    )
    return list(ds)


# ---- Multi-seed evaluation ---------------------------------------------------

def multi_seed_eval(
    dataset: List,
    cv_func,
    seeds: List[int] = None,
    verbose: bool = True,
    **cv_kwargs,
) -> Dict[str, Any]:
    """Run CV across multiple seeds and aggregate."""
    if seeds is None:
        seeds = SEEDS

    per_seed_aucs = []
    per_seed_f1s = []
    per_seed_results = {}

    for seed in seeds:
        if verbose:
            print(f"\n--- Seed {seed} ---")
        result = cv_func(dataset, seed=seed, verbose=verbose, **cv_kwargs)
        auc = result["aggregated"]["auc_mean"]
        f1 = result["aggregated"]["f1_mean"]
        per_seed_aucs.append(auc)
        per_seed_f1s.append(f1)
        per_seed_results[seed] = result
        if verbose:
            print(f"  Seed {seed}: AUC={auc:.3f}, F1={f1:.3f}")

    return {
        "per_seed": per_seed_results,
        "multi_seed": {
            "auc_mean": float(np.mean(per_seed_aucs)),
            "auc_std": float(np.std(per_seed_aucs)),
            "f1_mean": float(np.mean(per_seed_f1s)),
            "f1_std": float(np.std(per_seed_f1s)),
            "per_seed_aucs": {s: a for s, a in zip(seeds, per_seed_aucs)},
            "per_seed_f1s": {s: f for s, f in zip(seeds, per_seed_f1s)},
            "seeds": seeds,
        },
    }


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost position-only baseline on StatsBomb and tracking data",
    )
    parser.add_argument("--statsbomb", action="store_true", default=True,
                        help="Run StatsBomb freeze-frame experiment (default: True)")
    parser.add_argument("--no-statsbomb", action="store_true",
                        help="Skip StatsBomb experiment")
    parser.add_argument("--tracking-posonly", action="store_true",
                        help="Run tracking data position-only experiment")
    parser.add_argument("--permutation", action="store_true",
                        help="Run permutation tests")
    parser.add_argument("--all", action="store_true",
                        help="Run everything (StatsBomb + tracking + permutation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of folds for grouped k-fold CV (StatsBomb)")
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()

    if args.all:
        args.statsbomb = True
        args.tracking_posonly = True
        args.permutation = True
    if args.no_statsbomb:
        args.statsbomb = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    print(f"{'=' * 60}")
    print("XGBoost Position-Only Baseline")
    print(f"{'=' * 60}")
    print(f"Timestamp: {timestamp}")
    print(f"Seed: {args.seed}")
    print(f"Features: {len(POSITION_FEATURE_NAMES)} position-only")
    print(f"Feature names: {POSITION_FEATURE_NAMES}")

    sb_result = None
    tracking_result = None
    sb_perm = None
    tracking_perm = None

    # ---- StatsBomb freeze-frame experiment ------------------------------------
    if args.statsbomb:
        print(f"\n{'=' * 60}")
        print("StatsBomb Freeze-Frame: XGBoost Position-Only")
        print(f"{'=' * 60}")

        sb_dataset = load_statsbomb_dataset()
        n_corners = len(sb_dataset)
        n_matches = len(set(str(g.match_id) for g in sb_dataset))
        n_shots = sum(1 for g in sb_dataset if g.shot_label == 1)

        print(f"\nCorners: {n_corners}")
        print(f"Matches: {n_matches}")
        print(f"Shots: {n_shots}/{n_corners} ({100*n_shots/n_corners:.1f}%)")
        print(f"CV: grouped {args.n_folds}-fold")

        # Multi-seed evaluation
        print(f"\n--- Multi-seed evaluation ({len(SEEDS)} seeds) ---")
        sb_multi = multi_seed_eval(
            sb_dataset,
            grouped_kfold_xgboost,
            seeds=SEEDS,
            verbose=True,
            n_folds=args.n_folds,
        )

        sb_result = {
            "dataset": "statsbomb",
            "n_corners": n_corners,
            "n_matches": n_matches,
            "n_shots": n_shots,
            "timestamp": timestamp,
            **sb_multi,
        }

        # Save
        out_path = RESULTS_DIR / "statsbomb_xgboost_posonly.json"
        with open(out_path, "w") as f:
            json.dump(sb_result, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

        # Permutation test
        if args.permutation:
            print(f"\n--- StatsBomb permutation test (seed={args.seed}) ---")
            sb_perm = permutation_test_xgboost(
                sb_dataset,
                grouped_kfold_xgboost,
                n_permutations=args.n_permutations,
                seed=args.seed,
                verbose=True,
                n_folds=args.n_folds,
            )
            sb_perm["dataset"] = "statsbomb"
            sb_perm["n_corners"] = n_corners
            sb_perm["n_matches"] = n_matches
            sb_perm["timestamp"] = timestamp

            perm_path = RESULTS_DIR / "statsbomb_xgboost_posonly_perm.json"
            with open(perm_path, "w") as f:
                json.dump(sb_perm, f, indent=2)
            print(f"Saved: {perm_path}")

    # ---- Tracking position-only experiment ------------------------------------
    if args.tracking_posonly:
        print(f"\n{'=' * 60}")
        print("Tracking Data: XGBoost Position-Only (LOMO)")
        print(f"{'=' * 60}")

        tracking_dataset = load_tracking_dataset()
        n_corners = len(tracking_dataset)
        n_matches = len(get_match_ids(tracking_dataset))
        n_shots = sum(1 for g in tracking_dataset if g.shot_label == 1)

        print(f"\nCorners: {n_corners}")
        print(f"Matches: {n_matches}")
        print(f"Shots: {n_shots}/{n_corners} ({100*n_shots/n_corners:.1f}%)")
        print(f"CV: LOMO ({n_matches} folds)")

        # Multi-seed evaluation
        print(f"\n--- Multi-seed evaluation ({len(SEEDS)} seeds) ---")
        tracking_multi = multi_seed_eval(
            tracking_dataset,
            lomo_xgboost_posonly,
            seeds=SEEDS,
            verbose=True,
        )

        tracking_result = {
            "dataset": "tracking_combined",
            "n_corners": n_corners,
            "n_matches": n_matches,
            "n_shots": n_shots,
            "timestamp": timestamp,
            **tracking_multi,
        }

        # Save
        out_path = RESULTS_DIR / "tracking_xgboost_posonly.json"
        with open(out_path, "w") as f:
            json.dump(tracking_result, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

        # Permutation test
        if args.permutation:
            print(f"\n--- Tracking permutation test (seed={args.seed}) ---")
            tracking_perm = permutation_test_xgboost(
                tracking_dataset,
                lomo_xgboost_posonly,
                n_permutations=args.n_permutations,
                seed=args.seed,
                verbose=True,
            )
            tracking_perm["dataset"] = "tracking_combined"
            tracking_perm["n_corners"] = n_corners
            tracking_perm["n_matches"] = n_matches
            tracking_perm["timestamp"] = timestamp

            perm_path = RESULTS_DIR / "tracking_xgboost_posonly_perm.json"
            with open(perm_path, "w") as f:
                json.dump(tracking_perm, f, indent=2)
            print(f"Saved: {perm_path}")

    # ---- Final report ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    if sb_result:
        ms = sb_result["multi_seed"]
        print(f"\nStatsBomb XGBoost Position-Only Results")
        print(f"{'=' * 40}")
        print(f"Corners: {sb_result['n_corners']}")
        print(f"Matches: {sb_result['n_matches']}")
        print(f"Folds: {args.n_folds}")
        print(f"Features: 18 position-only")
        print(f"Multi-seed AUC: {ms['auc_mean']:.3f} +/- {ms['auc_std']:.3f}")
        print(f"Multi-seed F1:  {ms['f1_mean']:.3f} +/- {ms['f1_std']:.3f}")
        for seed, auc in ms["per_seed_aucs"].items():
            print(f"  Seed {seed}: AUC={auc:.3f}")
        if sb_perm:
            print(f"Permutation p-value: {sb_perm['p_value']:.3f}")

    if tracking_result:
        ms = tracking_result["multi_seed"]
        print(f"\nTracking XGBoost Position-Only Results")
        print(f"{'=' * 40}")
        print(f"Corners: {tracking_result['n_corners']}")
        print(f"Matches: {tracking_result['n_matches']}")
        print(f"Folds: {tracking_result['n_matches']} (LOMO)")
        print(f"Features: 18 position-only")
        print(f"Multi-seed AUC: {ms['auc_mean']:.3f} +/- {ms['auc_std']:.3f}")
        print(f"Multi-seed F1:  {ms['f1_mean']:.3f} +/- {ms['f1_std']:.3f}")
        for seed, auc in ms["per_seed_aucs"].items():
            print(f"  Seed {seed}: AUC={auc:.3f}")
        if tracking_perm:
            print(f"Permutation p-value: {tracking_perm['p_value']:.3f}")

    # Three-way comparison table
    print(f"\nThree-way comparison:")
    print(f"| XGBoost condition       | Data        | Features     | AUC   | p     |")
    print(f"|-------------------------|-------------|--------------|-------|-------|")
    print(f"| Tracking + velocity     | 143 SK+DFL  | 27 aggregate | 0.681 | 0.030 |")
    if tracking_result:
        ms = tracking_result["multi_seed"]
        p_str = f"{tracking_perm['p_value']:.3f}" if tracking_perm else "—"
        print(f"| Tracking pos-only      | {tracking_result['n_corners']} SK+DFL  "
              f"| 18 position  | {ms['auc_mean']:.3f} | {p_str} |")
    else:
        print(f"| Tracking pos-only      | —           | 18 position  | —     | —     |")
    if sb_result:
        ms = sb_result["multi_seed"]
        p_str = f"{sb_perm['p_value']:.3f}" if sb_perm else "—"
        print(f"| Freeze-frame pos-only  | {sb_result['n_corners']} SB      "
              f"| 18 position  | {ms['auc_mean']:.3f} | {p_str} |")
    else:
        print(f"| Freeze-frame pos-only  | —           | 18 position  | —     | —     |")


if __name__ == "__main__":
    main()
