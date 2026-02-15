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
    N_PERMUTATIONS,
    PRETRAINED_PATH,
    RESULTS_DIR,
    SEEDS,
)


def load_dataset(edge_type: str = "knn", k: int = 6):
    """Load the corner kick dataset."""
    from corner_prediction.data.dataset import CornerKickDataset

    dataset = CornerKickDataset(
        root=str(DATA_DIR),
        edge_type=edge_type,
        k=k,
    )
    print(f"Loaded {len(dataset)} graphs from {DATA_DIR}")
    return dataset


def load_records():
    """Load raw corner records (for edge_type rebuilding)."""
    records_path = DATA_DIR / "extracted_corners.pkl"
    with open(records_path, "rb") as f:
        return pickle.load(f)


def run_eval(args):
    """Run LOMO cross-validation."""
    from corner_prediction.training.evaluate import lomo_cv, save_results

    dataset = load_dataset()

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

    save_results(results, name=f"lomo_{args.mode}", output_dir=args.output_dir)
    return results


def run_permutation(args):
    """Run permutation tests."""
    from corner_prediction.training.evaluate import save_results
    from corner_prediction.training.permutation_test import (
        permutation_test_receiver,
        permutation_test_shot,
    )

    dataset = load_dataset()

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

    if args.permutation_target in ("receiver", "both"):
        recv_results = permutation_test_receiver(
            dataset,
            n_permutations=args.n_permutations,
            seed=args.seed,
            **lomo_kwargs,
        )
        save_results(recv_results, name="perm_receiver", output_dir=args.output_dir)

    if args.permutation_target in ("shot", "both"):
        shot_results = permutation_test_shot(
            dataset,
            n_permutations=args.n_permutations,
            seed=args.seed,
            receiver_mode="oracle",
            **lomo_kwargs,
        )
        save_results(shot_results, name="perm_shot", output_dir=args.output_dir)


def run_ablation(args):
    """Run ablation experiments."""
    from corner_prediction.training.ablation import (
        run_all_ablations,
        run_single_ablation,
    )
    from corner_prediction.training.evaluate import save_results

    dataset = load_dataset()
    records = load_records()

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

    if args.all_ablations:
        all_results = run_all_ablations(
            dataset, records=records, seed=args.seed,
            output_dir=args.output_dir, **lomo_kwargs,
        )
        save_results(all_results, name="ablation_all", output_dir=args.output_dir)
    elif args.ablation:
        results = run_single_ablation(
            args.ablation, dataset, records=records, seed=args.seed,
            **lomo_kwargs,
        )
        save_results(results, name=f"ablation_{args.ablation}", output_dir=args.output_dir)


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
    print(f"Seed: {args.seed}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if args.permutation_only:
        run_permutation(args)
    elif args.ablation or args.all_ablations:
        run_ablation(args)
    else:
        run_eval(args)

    print(f"\nDone! ({datetime.now()})")


if __name__ == "__main__":
    main()
