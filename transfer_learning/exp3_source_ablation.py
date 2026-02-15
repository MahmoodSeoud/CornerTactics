#!/usr/bin/env python3
"""
Experiment 3: Ablation by Data Source
======================================

Question: "Does noisy GSR data help or hurt?"

Conditions:
    S-A: Train on GSR only (2,080 train), eval on full test (459)
    S-B: Train on DFL+SkillCorner only (76 train), eval on full test (459)
    S-C: Train on all combined (2,156 train), eval on full test (459)

S-B uses frozen backbone + small head (same approach as Phase 3 with n=57)
since the training set is only 76 samples.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transfer_learning.multi_source_utils import (
    bootstrap_auc_ci,
    load_splits,
    prepare_pyg_data,
    run_training,
    save_results,
)


def run_exp3(
    seeds: list,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Experiment 3: Ablation by Data Source")
    print(f"  {datetime.now()}")
    print("=" * 70)

    train_raw, val_raw, test_raw = load_splits()

    # Split train/val by source
    gsr_train = [c for c in train_raw if c.get("source") == "soccernet_gsr"]
    gsr_val = [c for c in val_raw if c.get("source") == "soccernet_gsr"]
    hq_train = [c for c in train_raw if c.get("source") in ("dfl", "skillcorner")]
    hq_val = [c for c in val_raw if c.get("source") in ("dfl", "skillcorner")]

    gsr_shot = sum(1 for c in gsr_train if c["labels"]["shot_binary"] == 1)
    hq_shot = sum(1 for c in hq_train if c["labels"]["shot_binary"] == 1)

    print(f"  GSR train: {len(gsr_train)} ({gsr_shot} shots)")
    print(f"  GSR val:   {len(gsr_val)}")
    print(f"  HQ train:  {len(hq_train)} ({hq_shot} shots)")
    print(f"  HQ val:    {len(hq_val)}")
    print(f"  Full train: {len(train_raw)}")
    print(f"  Test:      {len(test_raw)}")

    all_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        test_data = prepare_pyg_data(test_raw)

        # S-A: GSR only
        model_a, result_a = run_training(
            prepare_pyg_data(gsr_train),
            prepare_pyg_data(gsr_val),
            test_data,
            pretrained=True,
            freeze_backbone=False,
            lr=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
            label=f"S-A GSR only (seed={seed})",
        )
        result_a["bootstrap_ci"] = bootstrap_auc_ci(result_a["test_metrics"], seed=seed)

        # S-B: DFL + SkillCorner only (small dataset → frozen backbone + small head)
        # If val set is too small for early stopping, use train loss instead
        hq_val_data = prepare_pyg_data(hq_val)
        if len(hq_val_data) < 5:
            # Use a subset of training data as pseudo-validation
            print("  WARNING: HQ val set too small, using last 20% of train as val")
            hq_all = hq_train.copy()
            np.random.seed(seed)
            np.random.shuffle(hq_all)
            split_idx = max(1, int(len(hq_all) * 0.8))
            hq_train_actual = hq_all[:split_idx]
            hq_val_actual = hq_all[split_idx:]
            hq_val_data = prepare_pyg_data(hq_val_actual)
            hq_train_data = prepare_pyg_data(hq_train_actual)
        else:
            hq_train_data = prepare_pyg_data(hq_train)

        model_b, result_b = run_training(
            hq_train_data,
            hq_val_data,
            test_data,
            pretrained=True,
            freeze_backbone=True,  # Frozen — only 76 train samples
            lr=1e-3,
            epochs=50,
            batch_size=8,
            patience=10,
            head_hidden=32,  # Small head for small dataset
            device=device,
            seed=seed,
            label=f"S-B DFL+SC only (seed={seed})",
        )
        result_b["bootstrap_ci"] = bootstrap_auc_ci(result_b["test_metrics"], seed=seed)

        # S-C: All combined
        model_c, result_c = run_training(
            prepare_pyg_data(train_raw),
            prepare_pyg_data(val_raw),
            test_data,
            pretrained=True,
            freeze_backbone=False,
            lr=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
            label=f"S-C All combined (seed={seed})",
        )
        result_c["bootstrap_ci"] = bootstrap_auc_ci(result_c["test_metrics"], seed=seed)

        all_results[seed] = {
            "S-A": result_a,
            "S-B": result_b,
            "S-C": result_c,
        }

    # Aggregate
    aggregated = {}
    for cond in ["S-A", "S-B", "S-C"]:
        aucs = [all_results[s][cond]["test_metrics"]["auc"] for s in seeds]
        f1s = [all_results[s][cond]["test_metrics"]["f1"] for s in seeds]
        accs = [all_results[s][cond]["test_metrics"]["accuracy"] for s in seeds]
        aggregated[cond] = {
            "test_auc_mean": float(np.mean(aucs)),
            "test_auc_std": float(np.std(aucs)),
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s)),
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "aucs": aucs,
        }

    # Print summary
    print(f"\n{'='*70}")
    print("  Experiment 3: Source Ablation Results")
    print(f"{'='*70}")
    labels = {
        "S-A": f"GSR only (n={len(gsr_train)})",
        "S-B": f"DFL+SC only (n={len(hq_train)})",
        "S-C": f"All combined (n={len(train_raw)})",
    }
    print(f"  {'Condition':<30} {'AUC':>14} {'F1':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    for cond in ["S-A", "S-B", "S-C"]:
        a = aggregated[cond]
        print(
            f"  {labels[cond]:<30} "
            f"{a['test_auc_mean']:.3f}±{a['test_auc_std']:.3f}   "
            f"{a['test_f1_mean']:.3f}±{a['test_f1_std']:.3f}"
        )
    print(f"{'='*70}")

    output = {
        "experiment": "exp3_source_ablation",
        "timestamp": str(datetime.now()),
        "data_sizes": {
            "gsr_train": len(gsr_train),
            "hq_train": len(hq_train),
            "full_train": len(train_raw),
            "test": len(test_raw),
        },
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    save_results(output, "exp3_source_ablation.pkl")
    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Source Ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_exp3(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
