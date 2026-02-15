"""Ablation experiments for corner kick prediction.

Runs the full pipeline under different feature configurations to isolate
which features contribute to prediction performance.

Ablation configs from two-stage-pipeline.md:
    position_only:    x, y, team, role (no velocity/detection)
    plus_velocity:    + vx, vy, speed
    plus_detection:   + is_detected (= all 13)
    full_features:    all 13 features (same as plus_detection, baseline)
    full_fc_edges:    all 13 features with fully connected edges
"""

import copy
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from corner_prediction.config import ABLATION_CONFIGS, RESULTS_DIR
from corner_prediction.data.build_graphs import build_graph_dataset
from corner_prediction.data.dataset import CornerKickDataset
from corner_prediction.training.evaluate import lomo_cv, print_results_table, save_results

logger = logging.getLogger(__name__)


def apply_feature_mask(dataset, active_features: List[int]) -> list:
    """Zero out inactive node features in the dataset.

    Creates deep copies to avoid modifying the original data.

    Args:
        dataset: Iterable of PyG Data objects (13 node features each).
        active_features: List of feature indices (0-12) to keep.

    Returns:
        New list of Data objects with inactive features zeroed.
    """
    all_indices = set(range(13))
    inactive = sorted(all_indices - set(active_features))

    if not inactive:
        # No masking needed, but still copy
        return [copy.deepcopy(g) for g in dataset]

    masked = []
    for g in dataset:
        g_copy = copy.deepcopy(g)
        for idx in inactive:
            g_copy.x[:, idx] = 0.0
        masked.append(g_copy)
    return masked


def rebuild_with_edge_type(records, edge_type: str, k: int = 6) -> list:
    """Rebuild graph dataset with different edge construction.

    Args:
        records: Raw corner records (from extracted_corners.pkl).
        edge_type: "knn" or "dense".
        k: KNN neighbor count.

    Returns:
        List of PyG Data objects.
    """
    return build_graph_dataset(records, edge_type=edge_type, k=k)


def run_single_ablation(
    ablation_name: str,
    dataset,
    records=None,
    seed: int = 42,
    verbose: bool = True,
    **lomo_kwargs,
) -> Dict[str, Any]:
    """Run one ablation configuration.

    Args:
        ablation_name: Key from ABLATION_CONFIGS.
        dataset: Original dataset (used for KNN-based ablations).
        records: Raw corner records (needed for edge_type changes).
        **lomo_kwargs: Passed through to lomo_cv.

    Returns:
        Dict with ablation name, config, and LOMO results.
    """
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name!r}. "
                         f"Choose from: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[ablation_name]

    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# Ablation: {ablation_name}")
        print(f"# {config['description']}")
        print(f"# Active features: {config['active_features']}")
        print(f"# Edge type: {config['edge_type']}")
        print(f"{'#' * 60}")

    # Handle edge type change (needs rebuilding from records)
    if config["edge_type"] == "dense":
        if records is None:
            raise ValueError(
                f"Ablation {ablation_name!r} requires edge_type='dense' but "
                f"records=None. Pass raw corner records to rebuild edges."
            )
        abl_dataset = rebuild_with_edge_type(records, "dense", config["k"])
    else:
        abl_dataset = list(dataset)

    # Apply feature mask
    abl_dataset = apply_feature_mask(abl_dataset, config["active_features"])

    # Run LOMO CV
    results = lomo_cv(abl_dataset, seed=seed, verbose=verbose, **lomo_kwargs)
    results["ablation"] = {
        "name": ablation_name,
        "config": config,
    }

    return results


def run_all_ablations(
    dataset,
    records=None,
    seed: int = 42,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    **lomo_kwargs,
) -> Dict[str, Any]:
    """Run all ablation configurations and produce comparison table.

    Returns:
        Dict mapping ablation_name → results.
    """
    all_results = {}

    for name in ABLATION_CONFIGS:
        results = run_single_ablation(
            name, dataset, records=records, seed=seed,
            verbose=verbose, **lomo_kwargs,
        )
        all_results[name] = results

        # Save intermediate results
        if output_dir:
            save_results(results, name=f"ablation_{name}", output_dir=output_dir)

    # Print comparison table
    if verbose:
        print_ablation_table(all_results)

    return all_results


def print_ablation_table(all_results: Dict[str, Any]) -> None:
    """Print thesis-ready ablation comparison table."""
    print(f"\n{'=' * 90}")
    print("Ablation Comparison")
    print(f"{'=' * 90}")
    header = (f"{'Ablation':<18} {'Recv Top1':>10} {'Recv Top3':>10} "
              f"{'Shot AUC (O)':>13} {'Shot AUC (P)':>13} {'Shot AUC (U)':>13}")
    print(header)
    print("-" * 90)

    for name, results in all_results.items():
        agg = results["aggregated"]
        r = agg["receiver"]
        so = agg["shot_oracle"]
        sp = agg["shot_predicted"]
        su = agg["shot_unconditional"]

        row = (f"{name:<18} "
               f"{r['top1_mean']:>6.3f}±{r['top1_std']:<4.3f}"
               f"{r['top3_mean']:>6.3f}±{r['top3_std']:<4.3f}"
               f"{so['auc_mean']:>7.3f}±{so['auc_std']:<5.3f}"
               f"{sp['auc_mean']:>7.3f}±{sp['auc_std']:<5.3f}"
               f"{su['auc_mean']:>7.3f}±{su['auc_std']:<5.3f}")
        print(row)

    # Random baselines
    print("-" * 90)
    print(f"{'Random baseline':<18} {'~0.147':>10} {'~0.441':>10} "
          f"{'0.500':>13} {'0.500':>13} {'0.500':>13}")
    print(f"{'=' * 90}")
