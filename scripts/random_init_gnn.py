#!/usr/bin/env python3
"""Run random-init GNN experiment: same USSF architecture, no pretrained weights.

Tests whether the USSF pretrained backbone is necessary, or whether a fresh GNN
can also find signal in tracking data.

Usage:
    # Multi-seed evaluation (5 seeds)
    python scripts/random_init_gnn.py --multi-seed --combined

    # Permutation test (100 perms, seed 42)
    python scripts/random_init_gnn.py --permutation --combined

    # Both (full experiment)
    python scripts/random_init_gnn.py --multi-seed --permutation --combined
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from corner_prediction.config import (
    BATCH_SIZE,
    DATA_DIR,
    N_PERMUTATIONS,
    RESULTS_DIR,
    SEEDS,
)
from corner_prediction.training.evaluate import lomo_cv, save_results


def load_dataset(combined: bool = True):
    """Load dataset with USSF-aligned features."""
    from corner_prediction.data.dataset import CornerKickDataset

    records_file = "combined_corners.pkl" if combined else "extracted_corners.pkl"
    dataset = CornerKickDataset(
        root=str(DATA_DIR),
        records_file=records_file,
        edge_type="dense",
        k=6,
        feature_mode="ussf_aligned",
    )
    label = "combined (SC + DFL)" if combined else "SkillCorner"
    print(f"Loaded {len(dataset)} graphs ({label} [ussf_aligned])")
    return dataset


def run_multi_seed(dataset, device, output_dir):
    """Run 5-seed evaluation with random-init backbone."""
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'#' * 60}")
        print(f"  SEED = {seed}")
        print(f"{'#' * 60}")

        results = lomo_cv(
            dataset,
            backbone_mode="ussf_random_init",
            pretrained_path=None,
            freeze=False,
            seed=seed,
            device=device,
            verbose=True,
            linear_heads=False,
        )

        save_results(results, name=f"combined_lomo_ussf_random_init_seed{seed}",
                     output_dir=output_dir)
        all_results[seed] = results

    # Print summary
    print(f"\n{'=' * 90}")
    print("MULTI-SEED SUMMARY (Random Init GNN)")
    print(f"{'=' * 90}")
    print(f"{'Seed':>6s} | {'Shot AUC (oracle)':>18s} | {'Shot AUC (pred)':>16s} | "
          f"{'Shot AUC (uncond)':>18s} | {'Recv Top-3':>11s}")
    print("-" * 90)

    oracle_aucs, pred_aucs, uncond_aucs, recv_top3s = [], [], [], []
    for seed in SEEDS:
        r = all_results[seed]
        a = r["aggregated"]
        o = a["shot_oracle"]["auc_mean"]
        p = a["shot_predicted"]["auc_mean"]
        u = a["shot_unconditional"]["auc_mean"]
        t3 = a["receiver"]["top3_mean"]
        oracle_aucs.append(o)
        pred_aucs.append(p)
        uncond_aucs.append(u)
        recv_top3s.append(t3)
        print(f"{seed:6d} | {o:18.3f} | {p:16.3f} | {u:18.3f} | {t3:11.3f}")

    print("-" * 90)
    print(f"{'Mean':>6s} | {np.mean(oracle_aucs):18.3f} | {np.mean(pred_aucs):16.3f} | "
          f"{np.mean(uncond_aucs):18.3f} | {np.mean(recv_top3s):11.3f}")
    print(f"{'Std':>6s} | {np.std(oracle_aucs):18.3f} | {np.std(pred_aucs):16.3f} | "
          f"{np.std(uncond_aucs):18.3f} | {np.std(recv_top3s):11.3f}")
    print(f"{'=' * 90}")

    # Per-fold breakdown for seed 42
    if 42 in all_results:
        print(f"\nPer-fold AUC (seed 42, oracle):")
        print(f"{'Fold':>4s} | {'Match':>12s} | {'n_test':>6s} | {'AUC':>6s}")
        print("-" * 40)
        for fold in all_results[42]["per_fold"]:
            print(f"{fold['fold_idx']+1:4d} | {fold['held_out_match']:>12s} | "
                  f"{fold['n_test']:6d} | {fold['shot_oracle']['auc']:.3f}")

    # Print param counts
    from corner_prediction.training.train import build_model
    model = build_model(
        backbone_mode="ussf_random_init",
        pretrained_path=None,
        freeze=False,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {total:,} total, {trainable:,} trainable")

    return all_results


def run_permutation(dataset, device, output_dir, n_permutations=100, seed=42):
    """Run permutation test for shot AUC with random-init backbone."""
    from corner_prediction.training.permutation_test import (
        permutation_test_shot,
    )

    result = permutation_test_shot(
        dataset,
        n_permutations=n_permutations,
        seed=seed,
        receiver_mode="oracle",
        backbone_mode="ussf_random_init",
        pretrained_path=None,
        freeze=False,
        device=device,
        linear_heads=False,
    )

    save_results(result, name="combined_perm_shot_ussf_random_init",
                 output_dir=output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Random-init GNN experiment")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run 5-seed LOMO evaluation")
    parser.add_argument("--permutation", action="store_true",
                        help="Run permutation test (100 perms, seed 42)")
    parser.add_argument("--combined", action="store_true",
                        help="Use combined dataset (143 corners)")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    if not args.multi_seed and not args.permutation:
        parser.error("Specify at least one of --multi-seed or --permutation")

    device = torch.device("cpu" if args.no_gpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"{'=' * 60}")
    print("Random-Init GNN Experiment")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {device}")
    print(f"Dataset: {'combined' if args.combined else 'SkillCorner only'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    dataset = load_dataset(combined=args.combined)
    start = time.time()

    if args.multi_seed:
        run_multi_seed(dataset, device, args.output_dir)

    if args.permutation:
        run_permutation(dataset, device, args.output_dir,
                        n_permutations=args.n_permutations, seed=args.seed)

    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTotal runtime: {int(hours)}h {int(mins)}m {int(secs)}s")
    print(f"Done! ({datetime.now()})")


if __name__ == "__main__":
    main()
