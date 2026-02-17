#!/usr/bin/env python3
"""Run baseline comparisons for corner kick prediction.

Usage:
    # Run all baselines
    python -m corner_prediction.baselines.run_baselines

    # Run specific baseline
    python -m corner_prediction.baselines.run_baselines --baseline random
    python -m corner_prediction.baselines.run_baselines --baseline heuristic
    python -m corner_prediction.baselines.run_baselines --baseline xgboost
    python -m corner_prediction.baselines.run_baselines --baseline mlp

    # Override output directory
    python -m corner_prediction.baselines.run_baselines --output-dir results/baselines
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from corner_prediction.config import DATA_DIR, RESULTS_DIR


def load_dataset(combined: bool = False):
    """Load the corner kick dataset.

    Args:
        combined: If True, load the combined SkillCorner + DFL dataset.
    """
    from corner_prediction.data.dataset import CornerKickDataset

    records_file = "combined_corners.pkl" if combined else "extracted_corners.pkl"
    dataset = CornerKickDataset(
        root=str(DATA_DIR), records_file=records_file, edge_type="knn", k=6,
    )
    label = "combined (SC + DFL)" if combined else "SkillCorner"
    print(f"Loaded {len(dataset)} graphs ({label}) from {DATA_DIR}")
    return dataset


def run_random(dataset, seed: int, output_dir: str, prefix: str = "") -> Dict:
    from corner_prediction.baselines.random_baseline import random_baseline_lomo
    from corner_prediction.training.evaluate import save_results

    results = random_baseline_lomo(dataset, seed=seed, verbose=True)
    save_results(results, name=f"{prefix}baseline_random", output_dir=output_dir)
    return results


def run_heuristic(dataset, seed: int, output_dir: str, prefix: str = "") -> Dict:
    from corner_prediction.baselines.heuristic_receiver import heuristic_receiver_lomo
    from corner_prediction.training.evaluate import save_results

    results = heuristic_receiver_lomo(dataset, seed=seed, verbose=True)
    save_results(results, name=f"{prefix}baseline_heuristic", output_dir=output_dir)
    return results


def run_xgboost(dataset, seed: int, output_dir: str, prefix: str = "") -> Dict:
    from corner_prediction.baselines.xgboost_baseline import xgboost_baseline_lomo
    from corner_prediction.training.evaluate import save_results

    results = xgboost_baseline_lomo(dataset, seed=seed, verbose=True)
    save_results(results, name=f"{prefix}baseline_xgboost", output_dir=output_dir)
    return results


def run_mlp(dataset, seed: int, output_dir: str, no_gpu: bool = False,
            prefix: str = "") -> Dict:
    from corner_prediction.baselines.mlp_baseline import mlp_baseline_lomo
    from corner_prediction.training.evaluate import save_results

    device = torch.device("cpu" if no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"MLP device: {device}")

    results = mlp_baseline_lomo(dataset, seed=seed, device=device, verbose=True)
    save_results(results, name=f"{prefix}baseline_mlp", output_dir=output_dir)
    return results


def print_baseline_comparison(all_results: Dict[str, Dict]) -> None:
    """Print comparison table across all baselines."""
    print(f"\n{'=' * 80}")
    print("Baseline Comparison Summary")
    print(f"{'=' * 80}")
    print(f"{'Model':<25s} {'Recv Top1':>10s} {'Recv Top3':>10s} "
          f"{'Shot AUC':>10s} {'Shot F1':>10s}")
    print(f"{'-' * 80}")

    for name, results in all_results.items():
        agg = results["aggregated"]
        r = agg["receiver"]
        s = agg["shot_oracle"]

        recv_top1 = (f"{r['top1_mean']:.3f}±{r['top1_std']:.2f}"
                     if r.get("n_folds", 0) > 0 else "—")
        recv_top3 = (f"{r['top3_mean']:.3f}±{r['top3_std']:.2f}"
                     if r.get("n_folds", 0) > 0 else "—")
        shot_auc = (f"{s['auc_mean']:.3f}±{s['auc_std']:.2f}"
                    if s.get("n_folds", 0) > 0 else "—")
        shot_f1 = (f"{s['f1_mean']:.3f}±{s['f1_std']:.2f}"
                   if s.get("n_folds", 0) > 0 else "—")

        print(f"{name:<25s} {recv_top1:>10s} {recv_top3:>10s} "
              f"{shot_auc:>10s} {shot_f1:>10s}")

    print(f"{'-' * 80}")
    print(f"{'Random baseline':<25s} {'~0.147':>10s} {'~0.441':>10s} "
          f"{'0.500':>10s} {'—':>10s}")
    print(f"{'=' * 80}")


BASELINE_RUNNERS = {
    "random": run_random,
    "heuristic": run_heuristic,
    "xgboost": run_xgboost,
}


def main():
    parser = argparse.ArgumentParser(
        description="Corner kick prediction: baseline comparisons",
    )
    parser.add_argument(
        "--baseline", type=str, default="all",
        choices=["random", "heuristic", "xgboost", "mlp", "all"],
        help="Which baseline to run (default: all)",
    )
    parser.add_argument("--combined", action="store_true",
                        help="Use combined SkillCorner + DFL dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"{'=' * 60}")
    print("Corner Kick Prediction — Baseline Comparisons")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    dataset = load_dataset(combined=args.combined)
    all_results = {}
    prefix = "combined_" if args.combined else ""

    if args.baseline == "all":
        baselines_to_run = ["random", "heuristic", "xgboost", "mlp"]
    else:
        baselines_to_run = [args.baseline]

    for name in baselines_to_run:
        print(f"\n{'#' * 60}")
        print(f"# Running baseline: {name}")
        print(f"{'#' * 60}")

        if name == "mlp":
            all_results[name] = run_mlp(
                dataset, args.seed, args.output_dir, no_gpu=args.no_gpu,
                prefix=prefix,
            )
        else:
            all_results[name] = BASELINE_RUNNERS[name](
                dataset, args.seed, args.output_dir, prefix=prefix,
            )

    if len(all_results) > 1:
        print_baseline_comparison(all_results)

    print(f"\nDone! ({datetime.now()})")


if __name__ == "__main__":
    main()
