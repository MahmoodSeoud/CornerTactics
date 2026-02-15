"""Baseline 1: Random prediction.

- Stage 1 (receiver): Uniform random over attacking outfield players.
- Stage 2 (shot): Predict with P(shot) = dataset shot rate.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score

from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.evaluate import (
    compute_receiver_metrics,
    compute_shot_metrics,
)

logger = logging.getLogger(__name__)

N_RANDOM_TRIALS = 1000  # Repeat random draws for stable estimates


def _random_receiver_fold(test_data: List, rng: np.random.RandomState) -> Dict:
    """Evaluate random receiver prediction on one fold's test data."""
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

        # Monte Carlo: repeat random draws and average
        top1_hits = 0
        top3_hits = 0
        for _ in range(N_RANDOM_TRIALS):
            perm = rng.permutation(candidate_indices)
            top1_hits += int(perm[0] == true_idx)
            top3_hits += int(true_idx in perm[:3])

        per_graph.append({
            "top1": top1_hits / N_RANDOM_TRIALS,
            "top3": top3_hits / N_RANDOM_TRIALS,
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


def _random_shot_fold(
    train_data: List, test_data: List, rng: np.random.RandomState,
) -> Dict:
    """Evaluate random shot prediction on one fold's test data.

    Predicts P(shot) = train set shot rate for all test samples.
    """
    # Compute shot rate from training data
    train_labels = np.array([g.shot_label for g in train_data], dtype=float)
    shot_rate = train_labels.mean() if len(train_labels) > 0 else 0.5

    test_labels = np.array([g.shot_label for g in test_data], dtype=float)
    n_samples = len(test_labels)

    # Constant prediction = shot_rate for all samples
    probs = np.full(n_samples, shot_rate)

    # AUC for constant predictor is 0.5 by definition
    auc = 0.5

    # F1 at optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(test_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_05 = (probs >= 0.5).astype(int)
    accuracy = (preds_05 == test_labels).mean()

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "accuracy": float(accuracy),
        "probs": probs.tolist(),
        "labels": test_labels.tolist(),
        "n_samples": n_samples,
        "n_positive": int(test_labels.sum()),
    }


def random_baseline_lomo(dataset, seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """Run random baseline with LOMO cross-validation.

    Returns results in the same format as lomo_cv().
    """
    rng = np.random.RandomState(seed)
    match_ids = get_match_ids(dataset)
    fold_results = []

    for fold_idx, held_out in enumerate(match_ids):
        if verbose:
            print(f"\n--- Random Baseline Fold {fold_idx + 1}/{len(match_ids)}: "
                  f"held_out={held_out} ---")

        train_data, test_data = lomo_split(dataset, held_out)
        if not test_data:
            continue

        receiver = _random_receiver_fold(test_data, rng)
        shot = _random_shot_fold(train_data, test_data, rng)

        fold_result = {
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(train_data),
            "n_val": 0,
            "n_test": len(test_data),
            "receiver": receiver,
            "shot_oracle": shot,
            "shot_predicted": shot,
            "shot_unconditional": shot,
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Receiver: top1={receiver['top1_acc']:.3f}, "
                  f"top3={receiver['top3_acc']:.3f} (n={receiver['n_labeled']})")
            print(f"  Shot: AUC={shot['auc']:.3f}, F1={shot['f1']:.3f}")

    agg_receiver = compute_receiver_metrics(fold_results)
    agg_shot = compute_shot_metrics(fold_results, "oracle")

    results = {
        "config": {
            "baseline": "random",
            "seed": seed,
            "n_random_trials": N_RANDOM_TRIALS,
            "n_folds": len(fold_results),
        },
        "per_fold": fold_results,
        "aggregated": {
            "receiver": agg_receiver,
            "shot_oracle": agg_shot,
            "shot_predicted": agg_shot,
            "shot_unconditional": agg_shot,
        },
    }

    if verbose:
        _print_random_results(results)

    return results


def _print_random_results(results: Dict) -> None:
    agg = results["aggregated"]
    r = agg["receiver"]
    s = agg["shot_oracle"]

    print(f"\n{'=' * 60}")
    print("Random Baseline Results")
    print(f"{'=' * 60}")
    print(f"Receiver Top-1: {r['top1_mean']:.3f} +/- {r['top1_std']:.3f}")
    print(f"Receiver Top-3: {r['top3_mean']:.3f} +/- {r['top3_std']:.3f}")
    print(f"Shot AUC:       {s['auc_mean']:.3f} +/- {s['auc_std']:.3f}")
    print(f"Shot F1:        {s['f1_mean']:.3f} +/- {s['f1_std']:.3f}")
    print(f"{'=' * 60}")
