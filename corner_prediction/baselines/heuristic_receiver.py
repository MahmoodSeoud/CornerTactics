"""Baseline 2: Heuristic receiver prediction.

Predict receiver = nearest attacking outfield player to goal center.
Goal center in normalized coordinates: (1.0, 0.0)
(attacking team attacks left-to-right toward x=+52.5).
"""

import logging
import math
from typing import Any, Dict, List

import numpy as np

from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.evaluate import compute_receiver_metrics

logger = logging.getLogger(__name__)

# Goal center in normalized coordinates (x/52.5, y/34.0)
GOAL_X = 1.0
GOAL_Y = 0.0


def _heuristic_receiver_fold(test_data: List) -> Dict:
    """Evaluate heuristic receiver prediction on one fold."""
    per_graph = []

    for g in test_data:
        if not g.has_receiver_label:
            continue

        mask = g.receiver_mask.numpy()
        label = g.receiver_label.numpy()
        candidate_indices = np.where(mask)[0]
        n_candidates = len(candidate_indices)

        if n_candidates == 0:
            continue

        true_idx = label.argmax()

        # Compute distance to goal center for each candidate
        x_coords = g.x[:, 0].numpy()  # feature 0: x_norm
        y_coords = g.x[:, 1].numpy()  # feature 1: y_norm

        distances = []
        for idx in candidate_indices:
            dx = x_coords[idx] - GOAL_X
            dy = y_coords[idx] - GOAL_Y
            distances.append(math.sqrt(dx * dx + dy * dy))

        distances = np.array(distances)

        # Rank by distance (ascending = nearest first)
        ranked_order = np.argsort(distances)
        ranked_candidates = candidate_indices[ranked_order]

        top1 = ranked_candidates[0] == true_idx
        top3 = true_idx in ranked_candidates[:3]

        per_graph.append({
            "top1": bool(top1),
            "top3": bool(top3),
            "n_candidates": n_candidates,
        })

    n_labeled = len(per_graph)
    if n_labeled == 0:
        return {"top1_acc": 0.0, "top3_acc": 0.0, "n_labeled": 0, "per_graph": []}

    top1_acc = sum(g["top1"] for g in per_graph) / n_labeled
    top3_acc = sum(g["top3"] for g in per_graph) / n_labeled

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "n_labeled": n_labeled,
        "per_graph": per_graph,
    }


def heuristic_receiver_lomo(
    dataset, seed: int = 42, verbose: bool = True,
) -> Dict[str, Any]:
    """Run heuristic receiver baseline with LOMO cross-validation.

    Returns results in same format as lomo_cv(), but only receiver metrics
    are meaningful. Shot metrics are set to dummy values.
    """
    match_ids = get_match_ids(dataset)
    fold_results = []

    dummy_shot = {
        "auc": 0.5, "f1": 0.0, "f1_threshold": 0.5,
        "accuracy": 0.0, "probs": [], "labels": [],
        "n_samples": 0, "n_positive": 0,
    }

    for fold_idx, held_out in enumerate(match_ids):
        if verbose:
            print(f"\n--- Heuristic Baseline Fold {fold_idx + 1}/{len(match_ids)}: "
                  f"held_out={held_out} ---")

        _, test_data = lomo_split(dataset, held_out)
        if not test_data:
            continue

        receiver = _heuristic_receiver_fold(test_data)

        fold_result = {
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": 0,
            "n_val": 0,
            "n_test": len(test_data),
            "receiver": receiver,
            "shot_oracle": dummy_shot,
            "shot_predicted": dummy_shot,
            "shot_unconditional": dummy_shot,
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Receiver: top1={receiver['top1_acc']:.3f}, "
                  f"top3={receiver['top3_acc']:.3f} (n={receiver['n_labeled']})")

    agg_receiver = compute_receiver_metrics(fold_results)

    results = {
        "config": {
            "baseline": "heuristic_receiver",
            "seed": seed,
            "goal_center": (GOAL_X, GOAL_Y),
            "n_folds": len(fold_results),
        },
        "per_fold": fold_results,
        "aggregated": {
            "receiver": agg_receiver,
            "shot_oracle": {"auc_mean": 0.5, "auc_std": 0.0, "f1_mean": 0.0,
                            "f1_std": 0.0, "acc_mean": 0.0, "acc_std": 0.0,
                            "n_folds": 0, "per_fold_auc": [], "per_fold_f1": []},
            "shot_predicted": {"auc_mean": 0.5, "auc_std": 0.0, "f1_mean": 0.0,
                               "f1_std": 0.0, "acc_mean": 0.0, "acc_std": 0.0,
                               "n_folds": 0, "per_fold_auc": [], "per_fold_f1": []},
            "shot_unconditional": {"auc_mean": 0.5, "auc_std": 0.0, "f1_mean": 0.0,
                                   "f1_std": 0.0, "acc_mean": 0.0, "acc_std": 0.0,
                                   "n_folds": 0, "per_fold_auc": [], "per_fold_f1": []},
        },
    }

    if verbose:
        _print_heuristic_results(results)

    return results


def _print_heuristic_results(results: Dict) -> None:
    r = results["aggregated"]["receiver"]
    print(f"\n{'=' * 60}")
    print("Heuristic Receiver Baseline Results")
    print(f"{'=' * 60}")
    print(f"Method: Nearest attacking outfield player to goal center")
    print(f"Receiver Top-1: {r['top1_mean']:.3f} +/- {r['top1_std']:.3f}")
    print(f"Receiver Top-3: {r['top3_mean']:.3f} +/- {r['top3_std']:.3f}")
    print(f"{'=' * 60}")
