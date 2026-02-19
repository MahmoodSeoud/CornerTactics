#!/usr/bin/env python3
"""Single entry point for the corner kick prediction pipeline.

Usage:
    # Full LOMO evaluation (pretrained backbone)
    python -m corner_prediction.run_all

    # From-scratch backbone
    python -m corner_prediction.run_all --mode scratch

    # Permutation tests only
    python -m corner_prediction.run_all --permutation-only --n-permutations 100

    # Single ablation
    python -m corner_prediction.run_all --ablation position_only

    # All ablations
    python -m corner_prediction.run_all --all-ablations

    # DFL Integration (Task 7): extract DFL corners, merge, and evaluate
    python -m corner_prediction.run_all --extract-dfl --combined

    # Combined dataset evaluation (after DFL extraction is done)
    python -m corner_prediction.run_all --combined
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from corner_prediction.config import (
    ABLATION_CONFIGS,
    BATCH_SIZE,
    DATA_DIR,
    DFL_DATA_DIR,
    N_PERMUTATIONS,
    PRETRAINED_PATH,
    RESULTS_DIR,
    SEEDS,
)


def load_dataset(edge_type: str = "knn", k: int = 6, combined: bool = False):
    """Load the corner kick dataset.

    Args:
        edge_type: "knn" or "dense".
        k: Number of KNN neighbors.
        combined: If True, load the combined SkillCorner + DFL dataset.
    """
    from corner_prediction.data.dataset import CornerKickDataset

    records_file = "combined_corners.pkl" if combined else "extracted_corners.pkl"
    dataset = CornerKickDataset(
        root=str(DATA_DIR),
        records_file=records_file,
        edge_type=edge_type,
        k=k,
    )
    label = "combined (SC + DFL)" if combined else "SkillCorner"
    print(f"Loaded {len(dataset)} graphs ({label}) from {DATA_DIR}")
    return dataset


def load_records(combined: bool = False):
    """Load raw corner records (for edge_type rebuilding)."""
    fname = "combined_corners.pkl" if combined else "extracted_corners.pkl"
    records_path = DATA_DIR / fname
    with open(records_path, "rb") as f:
        return pickle.load(f)


def run_extract_dfl():
    """Extract DFL corners and merge with SkillCorner dataset."""
    from corner_prediction.data.extract_dfl_corners import extract_all_dfl_corners
    from corner_prediction.data.merge_datasets import merge_records

    # Step 1: Extract DFL corners from raw tracking data
    print("\n--- Step 1: Extracting DFL corners ---")
    dfl_records = extract_all_dfl_corners(data_dir=DFL_DATA_DIR)
    print(f"Extracted {len(dfl_records)} DFL corners")

    # Save DFL records separately
    dfl_path = DATA_DIR / "dfl_extracted_corners.pkl"
    with open(dfl_path, "wb") as f:
        pickle.dump(dfl_records, f)
    print(f"Saved DFL records to {dfl_path}")

    # Step 2: Load SkillCorner records and merge
    print("\n--- Step 2: Merging datasets ---")
    sc_path = DATA_DIR / "extracted_corners.pkl"
    with open(sc_path, "rb") as f:
        sc_records = pickle.load(f)
    print(f"Loaded {len(sc_records)} SkillCorner records")

    combined = merge_records(sc_records, dfl_records)

    # Step 3: Save combined dataset
    combined_path = DATA_DIR / "combined_corners.pkl"
    with open(combined_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"Saved {len(combined)} combined records to {combined_path}")

    return combined


def run_eval(args):
    """Run LOMO cross-validation."""
    from corner_prediction.training.evaluate import lomo_cv, save_results

    dataset = load_dataset(combined=args.combined)

    pretrained_path = PRETRAINED_PATH if args.mode == "pretrained" else None
    if pretrained_path and not pretrained_path.exists():
        print(f"WARNING: Pretrained weights not found at {pretrained_path}")
        print("Falling back to scratch mode")
        args.mode = "scratch"
        pretrained_path = None

    device = torch.device("cpu" if args.no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    results = lomo_cv(
        dataset,
        backbone_mode=args.mode,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        freeze=(args.mode == "pretrained"),
        seed=args.seed,
        device=device,
        verbose=True,
    )

    prefix = "combined_" if args.combined else ""
    save_results(results, name=f"{prefix}lomo_{args.mode}", output_dir=args.output_dir)
    return results


def run_permutation(args):
    """Run permutation tests."""
    from corner_prediction.training.evaluate import save_results
    from corner_prediction.training.permutation_test import (
        permutation_test_receiver,
        permutation_test_shot,
    )

    dataset = load_dataset(combined=args.combined)

    pretrained_path = PRETRAINED_PATH if args.mode == "pretrained" else None
    if pretrained_path and not pretrained_path.exists():
        args.mode = "scratch"
        pretrained_path = None

    device = torch.device("cpu" if args.no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    lomo_kwargs = dict(
        backbone_mode=args.mode,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        freeze=(args.mode == "pretrained"),
        device=device,
    )

    prefix = "combined_" if args.combined else ""

    if args.permutation_target in ("receiver", "both"):
        recv_results = permutation_test_receiver(
            dataset,
            n_permutations=args.n_permutations,
            seed=args.seed,
            **lomo_kwargs,
        )
        save_results(recv_results, name=f"{prefix}perm_receiver", output_dir=args.output_dir)

    if args.permutation_target in ("shot", "both"):
        shot_results = permutation_test_shot(
            dataset,
            n_permutations=args.n_permutations,
            seed=args.seed,
            receiver_mode="oracle",
            **lomo_kwargs,
        )
        save_results(shot_results, name=f"{prefix}perm_shot", output_dir=args.output_dir)


def run_ablation(args):
    """Run ablation experiments."""
    from corner_prediction.training.ablation import (
        run_all_ablations,
        run_single_ablation,
    )
    from corner_prediction.training.evaluate import save_results

    dataset = load_dataset(combined=args.combined)
    records = load_records(combined=args.combined)

    pretrained_path = PRETRAINED_PATH if args.mode == "pretrained" else None
    if pretrained_path and not pretrained_path.exists():
        args.mode = "scratch"
        pretrained_path = None

    device = torch.device("cpu" if args.no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    lomo_kwargs = dict(
        backbone_mode=args.mode,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        freeze=(args.mode == "pretrained"),
        device=device,
    )

    prefix = "combined_" if args.combined else ""

    if args.all_ablations:
        all_results = run_all_ablations(
            dataset, records=records, seed=args.seed,
            output_dir=args.output_dir, **lomo_kwargs,
        )
        save_results(all_results, name=f"{prefix}ablation_all", output_dir=args.output_dir)
    elif args.ablation:
        results = run_single_ablation(
            args.ablation, dataset, records=records, seed=args.seed,
            **lomo_kwargs,
        )
        save_results(results, name=f"{prefix}ablation_{args.ablation}", output_dir=args.output_dir)


def run_baselines(args):
    """Run baseline comparisons."""
    from corner_prediction.baselines.run_baselines import (
        load_dataset as load_bl_dataset,
        run_heuristic,
        run_mlp,
        run_random,
        run_xgboost,
        print_baseline_comparison,
    )

    dataset = load_bl_dataset(combined=args.combined)
    all_results = {}
    prefix = "combined_" if args.combined else ""

    if args.baselines == "all":
        baselines_to_run = ["random", "heuristic", "xgboost", "mlp"]
    else:
        baselines_to_run = [args.baselines]

    runners = {
        "random": lambda ds: run_random(ds, args.seed, args.output_dir, prefix=prefix),
        "heuristic": lambda ds: run_heuristic(ds, args.seed, args.output_dir, prefix=prefix),
        "xgboost": lambda ds: run_xgboost(ds, args.seed, args.output_dir, prefix=prefix),
        "mlp": lambda ds: run_mlp(ds, args.seed, args.output_dir, no_gpu=args.no_gpu, prefix=prefix),
    }

    for name in baselines_to_run:
        all_results[name] = runners[name](dataset)

    if len(all_results) > 1:
        print_baseline_comparison(all_results)


def run_baseline_permutation(args):
    """Run permutation tests on baseline models."""
    from corner_prediction.baselines.permutation_test_baselines import (
        permutation_test_baseline,
    )
    from corner_prediction.training.evaluate import save_results

    dataset = load_dataset(combined=args.combined)
    prefix = "combined_" if args.combined else ""

    targets = (
        ["mlp", "xgboost"] if args.baseline_permutation == "both"
        else [args.baseline_permutation]
    )

    device = torch.device("cpu" if args.no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    for name in targets:
        if name == "mlp":
            from corner_prediction.baselines.mlp_baseline import mlp_baseline_lomo
            result = permutation_test_baseline(
                dataset,
                baseline_fn=mlp_baseline_lomo,
                baseline_name="mlp",
                n_permutations=args.n_permutations,
                seed=args.seed,
                verbose=True,
                device=device,
            )
        elif name == "xgboost":
            from corner_prediction.baselines.xgboost_baseline import (
                xgboost_baseline_lomo,
            )
            result = permutation_test_baseline(
                dataset,
                baseline_fn=xgboost_baseline_lomo,
                baseline_name="xgboost",
                n_permutations=args.n_permutations,
                seed=args.seed,
                verbose=True,
            )

        save_results(result, name=f"{prefix}perm_{name}", output_dir=args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Corner kick prediction: training & evaluation pipeline",
    )

    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--eval-only", action="store_true",
                       help="Run LOMO evaluation only (default)")
    group.add_argument("--permutation-only", action="store_true",
                       help="Run permutation tests only")
    group.add_argument("--ablation", type=str, default=None,
                       choices=list(ABLATION_CONFIGS.keys()),
                       help="Run a single ablation")
    group.add_argument("--all-ablations", action="store_true",
                       help="Run all ablation configs")
    group.add_argument("--baselines", type=str, default=None,
                       choices=["random", "heuristic", "xgboost", "mlp", "all"],
                       help="Run baseline comparisons")
    group.add_argument("--baseline-permutation", type=str, default=None,
                       choices=["mlp", "xgboost", "both"],
                       help="Run permutation tests on baseline models")
    group.add_argument("--visualize", action="store_true",
                       help="Generate all thesis-ready figures from results")

    # DFL integration (Task 7)
    parser.add_argument("--combined", action="store_true",
                        help="Use combined SkillCorner + DFL dataset")
    parser.add_argument("--extract-dfl", action="store_true",
                        help="Extract DFL corners from raw data before evaluation")

    # Model config
    parser.add_argument("--mode", choices=["pretrained", "scratch"],
                        default="pretrained",
                        help="Backbone mode (default: pretrained)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true")

    # Permutation config
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--permutation-target", choices=["receiver", "shot", "both"],
                        default="both")

    # Output
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"{'=' * 60}")
    print("Corner Kick Prediction Pipeline")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {'combined (SC + DFL)' if args.combined else 'SkillCorner only'}")
    print(f"Seed: {args.seed}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    # DFL extraction (if requested)
    if args.extract_dfl:
        run_extract_dfl()
        if not args.combined:
            print("NOTE: --extract-dfl implies --combined, enabling combined mode")
            args.combined = True

    # Determine which pipeline step to run
    has_explicit_mode = (
        args.eval_only or args.permutation_only or args.ablation
        or args.all_ablations or args.baselines or args.baseline_permutation
        or args.visualize
    )

    if args.permutation_only:
        run_permutation(args)
    elif args.ablation or args.all_ablations:
        run_ablation(args)
    elif args.baselines:
        run_baselines(args)
    elif args.baseline_permutation:
        run_baseline_permutation(args)
    elif args.visualize:
        from corner_prediction.visualization.generate_all import generate_all
        results_dir = Path(args.output_dir)
        figures_dir = results_dir / "figures"
        generate_all(results_dir, figures_dir, show=False)
    elif args.extract_dfl and not has_explicit_mode:
        # Extract-only: don't fall through to eval
        print("DFL extraction complete. Use --eval-only --combined to evaluate.")
    else:
        run_eval(args)

    print(f"\nDone! ({datetime.now()})")


if __name__ == "__main__":
    main()
