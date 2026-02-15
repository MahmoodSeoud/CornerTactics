#!/usr/bin/env python3
"""Generate all thesis-ready figures from corner kick prediction results.

Usage:
    python -m corner_prediction.visualization.generate_all
    python -m corner_prediction.visualization.generate_all --output-dir results/figures
    python -m corner_prediction.visualization.generate_all --show
"""

import argparse
import glob
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from corner_prediction.config import DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def find_latest_result(results_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent result file matching a prefix.

    Looks for files like {prefix}_YYYYMMDD_HHMMSS.pkl and returns
    the one with the latest timestamp.
    """
    pattern = str(results_dir / f"{prefix}_*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])  # sorted alphabetically = latest timestamp


def load_pickle(path: Path) -> Optional[dict]:
    """Load a pickle file, returning None on failure."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def load_all_results(results_dir: Path) -> Dict[str, Optional[dict]]:
    """Load all available result files from the results directory.

    Returns dict with keys: lomo_pretrained, lomo_scratch, ablation_all,
    perm_receiver, perm_shot, and any baseline_* results.
    """
    results = {}

    # LOMO results
    for mode in ("pretrained", "scratch"):
        path = find_latest_result(results_dir, f"lomo_{mode}")
        if path:
            results[f"lomo_{mode}"] = load_pickle(path)
            logger.info("Loaded %s from %s", f"lomo_{mode}", path.name)

    # Ablation results (prefer ablation_all over individual)
    path = find_latest_result(results_dir, "ablation_all")
    if path:
        results["ablation_all"] = load_pickle(path)
        logger.info("Loaded ablation_all from %s", path.name)

    # Permutation tests
    for target in ("receiver", "shot"):
        path = find_latest_result(results_dir, f"perm_{target}")
        if path:
            results[f"perm_{target}"] = load_pickle(path)
            logger.info("Loaded perm_%s from %s", target, path.name)

    # Baselines
    baselines = {}
    for name in ("random", "heuristic", "xgboost", "mlp"):
        path = find_latest_result(results_dir, f"baseline_{name}")
        if path:
            baselines[name] = load_pickle(path)
            logger.info("Loaded baseline_%s from %s", name, path.name)
    if baselines:
        results["baselines"] = baselines

    return results


def load_dataset():
    """Load the corner kick PyG dataset (needed for Figure 5)."""
    try:
        from corner_prediction.data.dataset import CornerKickDataset
        dataset = CornerKickDataset(root=str(DATA_DIR), edge_type="knn", k=6)
        return dataset
    except Exception as e:
        logger.warning("Could not load dataset: %s", e)
        return None


def generate_all(
    results_dir: Path,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Generate all thesis figures from available results.

    Gracefully skips figures when required data is missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("Corner Kick Prediction: Figure Generation")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")

    # Load all available results
    all_results = load_all_results(results_dir)
    lomo = all_results.get("lomo_pretrained")
    ablation = all_results.get("ablation_all")
    perm_recv = all_results.get("perm_receiver")
    perm_shot = all_results.get("perm_shot")
    baselines = all_results.get("baselines")

    print(f"\nAvailable results: {list(all_results.keys())}")

    generated = []

    # Figure 1: Receiver prediction example
    print("\n--- Figure 1: Receiver Prediction Example ---")
    from corner_prediction.visualization.plot_corner import generate as gen_corner
    path = gen_corner(output_dir, show=show)
    if path:
        generated.append(path)

    # Figure 2: Ablation comparison
    print("\n--- Figure 2: Ablation Comparison ---")
    from corner_prediction.visualization.plot_ablation import generate as gen_ablation
    path = gen_ablation(output_dir, ablation_results=ablation, show=show)
    if path:
        generated.append(path)

    # Figure 3: Shot AUC comparison
    print("\n--- Figure 3: Shot AUC Comparison ---")
    from corner_prediction.visualization.plot_shot_auc import generate as gen_shot
    path = gen_shot(output_dir, lomo_results=lomo,
                    ablation_results=ablation,
                    baseline_results=baselines, show=show)
    if path:
        generated.append(path)

    # Figure 4: Two-stage benefit
    print("\n--- Figure 4: Two-Stage Benefit ---")
    from corner_prediction.visualization.plot_two_stage import generate as gen_two_stage
    path = gen_two_stage(output_dir, lomo_results=lomo, show=show)
    if path:
        generated.append(path)

    # Figure 5: Detection rate sensitivity
    print("\n--- Figure 5: Detection Rate Sensitivity ---")
    dataset = load_dataset()
    from corner_prediction.visualization.plot_sensitivity import generate as gen_sens
    path = gen_sens(output_dir, lomo_results=lomo, dataset=dataset, show=show)
    if path:
        generated.append(path)

    # Bonus: Permutation tests
    print("\n--- Bonus: Permutation Tests ---")
    from corner_prediction.visualization.plot_permutation import generate as gen_perm
    path = gen_perm(output_dir, perm_receiver=perm_recv, perm_shot=perm_shot,
                    show=show)
    if path:
        generated.append(path)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated)} figures:")
    for p in generated:
        print(f"  {p}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready figures from prediction results",
    )
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="Directory containing result pickle files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results_dir/figures)")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"

    generate_all(results_dir, output_dir, show=args.show)
    print(f"\nDone! ({datetime.now()})")


if __name__ == "__main__":
    main()
