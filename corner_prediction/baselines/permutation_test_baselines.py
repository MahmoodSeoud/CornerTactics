"""Permutation tests for baseline models (MLP, XGBoost).

Reuses shuffle_shot_labels from the GNN permutation module so that the
label-shuffling logic is identical across all models.  Accepts any
baseline_lomo_fn callable that follows the same signature as
mlp_baseline_lomo / xgboost_baseline_lomo.
"""

import logging
from typing import Any, Callable, Dict

import numpy as np

from corner_prediction.config import N_PERMUTATIONS
from corner_prediction.training.permutation_test import shuffle_shot_labels

logger = logging.getLogger(__name__)


def permutation_test_baseline(
    dataset,
    baseline_fn: Callable,
    baseline_name: str = "baseline",
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
    verbose: bool = True,
    **baseline_kwargs,
) -> Dict[str, Any]:
    """Permutation test for a baseline model's shot AUC.

    Args:
        dataset: List of graph objects with .shot_label attributes.
        baseline_fn: LOMO evaluation function (e.g. mlp_baseline_lomo).
            Must accept (dataset, seed=, verbose=, **kwargs) and return a
            dict with ["aggregated"]["shot_oracle"]["auc_mean"].
        baseline_name: Human-readable name for logging.
        n_permutations: Number of label shuffles.
        seed: Random seed for reproducibility.
        verbose: Print progress.
        **baseline_kwargs: Extra kwargs forwarded to baseline_fn
            (e.g. device for MLP).

    Returns:
        Dict matching GNN permutation test output format.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: {baseline_name} Shot AUC (n={n_permutations})")
        print(f"{'=' * 60}")

    # --- Real metric ---
    if verbose:
        print("Computing real metric...")
    real_results = baseline_fn(dataset, seed=seed, verbose=False, **baseline_kwargs)
    real_metric = real_results["aggregated"]["shot_oracle"]["auc_mean"]

    if verbose:
        print(f"Real AUC: {real_metric:.4f}")

    # --- Null distribution ---
    null_metrics = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_results = baseline_fn(
            shuffled, seed=seed, verbose=False, **baseline_kwargs,
        )
        null_metric = perm_results["aggregated"]["shot_oracle"]["auc_mean"]
        null_metrics.append(null_metric)

    null_metrics = np.array(null_metrics)
    p_value = (np.sum(null_metrics >= real_metric) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_metrics.mean():.4f}, "
              f"std={null_metrics.std():.4f}")
        print(f"Null range: [{null_metrics.min():.4f}, {null_metrics.max():.4f}]")
        print(f"p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return {
        "metric": f"shot_auc_{baseline_name}",
        "baseline": baseline_name,
        "real_metric": float(real_metric),
        "null_distribution": null_metrics.tolist(),
        "null_mean": float(null_metrics.mean()),
        "null_std": float(null_metrics.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }
