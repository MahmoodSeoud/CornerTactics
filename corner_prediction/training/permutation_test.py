"""Permutation tests for statistical validation.

Shuffles labels N times, retrains, and compares real metrics against
the null distribution to compute p-values.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from corner_prediction.config import N_PERMUTATIONS
from corner_prediction.training.evaluate import lomo_cv

logger = logging.getLogger(__name__)


def shuffle_receiver_labels(dataset, rng: np.random.RandomState) -> list:
    """Shuffle receiver labels within each graph's valid candidates.

    For each graph with a receiver label, randomly reassign it to one of the
    masked (valid candidate) nodes. Preserves mask structure and overall
    label rate.
    """
    shuffled = []
    for g in dataset:
        g_copy = copy.deepcopy(g)
        if g_copy.has_receiver_label:
            mask = g_copy.receiver_mask
            candidate_indices = mask.nonzero(as_tuple=True)[0]
            if len(candidate_indices) > 0:
                new_label = torch.zeros_like(g_copy.receiver_label)
                chosen = candidate_indices[rng.randint(len(candidate_indices))]
                new_label[chosen] = 1.0
                g_copy.receiver_label = new_label
        shuffled.append(g_copy)
    return shuffled


def shuffle_shot_labels(dataset, rng: np.random.RandomState) -> list:
    """Shuffle shot labels across all graphs. Preserves positive rate."""
    labels = [g.shot_label for g in dataset]
    rng.shuffle(labels)

    shuffled = []
    for g, new_label in zip(dataset, labels):
        g_copy = copy.deepcopy(g)
        g_copy.shot_label = new_label
        shuffled.append(g_copy)
    return shuffled


def permutation_test_receiver(
    dataset,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
    verbose: bool = True,
    **lomo_kwargs,
) -> Dict[str, Any]:
    """Permutation test for receiver prediction (top-3 accuracy).

    Returns:
        Dict with real_metric, null_distribution, p_value.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: Receiver (n={n_permutations})")
        print(f"{'=' * 60}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_results = lomo_cv(dataset, seed=seed, verbose=False, **lomo_kwargs)
    real_metric = real_results["aggregated"]["receiver"]["top3_mean"]

    if verbose:
        print(f"Real top-3 accuracy: {real_metric:.4f}")

    # Null distribution
    null_metrics = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_receiver_labels(dataset, rng)
        perm_results = lomo_cv(shuffled, seed=seed, verbose=False, **lomo_kwargs)
        null_metric = perm_results["aggregated"]["receiver"]["top3_mean"]
        null_metrics.append(null_metric)

    # p-value: fraction of null >= real (plus 1 for continuity correction)
    null_metrics = np.array(null_metrics)
    p_value = (np.sum(null_metrics >= real_metric) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_metrics.mean():.4f}, "
              f"std={null_metrics.std():.4f}")
        print(f"p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return {
        "metric": "receiver_top3_acc",
        "real_metric": float(real_metric),
        "null_distribution": null_metrics.tolist(),
        "null_mean": float(null_metrics.mean()),
        "null_std": float(null_metrics.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }


def permutation_test_shot(
    dataset,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
    receiver_mode: str = "oracle",
    verbose: bool = True,
    **lomo_kwargs,
) -> Dict[str, Any]:
    """Permutation test for shot prediction (AUC).

    Args:
        receiver_mode: Which receiver conditioning to test. Default "oracle".

    Returns:
        Dict with real_metric, null_distribution, p_value.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: Shot AUC ({receiver_mode}, n={n_permutations})")
        print(f"{'=' * 60}")

    # Map receiver_mode to the key used in fold results
    # eval_shot takes "none" but fold results store as "unconditional"
    mode_to_key = {"oracle": "shot_oracle", "predicted": "shot_predicted", "none": "shot_unconditional"}
    shot_key = mode_to_key.get(receiver_mode, f"shot_{receiver_mode}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_results = lomo_cv(dataset, seed=seed, verbose=False, **lomo_kwargs)
    real_metric = real_results["aggregated"][shot_key]["auc_mean"]

    if verbose:
        print(f"Real AUC ({receiver_mode}): {real_metric:.4f}")

    # Null distribution
    null_metrics = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_results = lomo_cv(shuffled, seed=seed, verbose=False, **lomo_kwargs)
        null_metric = perm_results["aggregated"][shot_key]["auc_mean"]
        null_metrics.append(null_metric)

    null_metrics = np.array(null_metrics)
    p_value = (np.sum(null_metrics >= real_metric) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_metrics.mean():.4f}, "
              f"std={null_metrics.std():.4f}")
        print(f"p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return {
        "metric": f"shot_auc_{receiver_mode}",
        "real_metric": float(real_metric),
        "null_distribution": null_metrics.tolist(),
        "null_mean": float(null_metrics.mean()),
        "null_std": float(null_metrics.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }
