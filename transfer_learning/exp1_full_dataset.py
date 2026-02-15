#!/usr/bin/env python3
"""
Experiment 1: Full Multi-Source Dataset
========================================

Question: "Does 50x more data solve the corner prediction problem?"

Conditions:
    MS-A: USSF pretrained + frozen backbone (linear probe)
    MS-B: USSF pretrained + fine-tuned (unfrozen)
    MS-C: Random init (train from scratch)
    MS-D: Majority class baseline

Reports AUC, F1, bootstrap 95% CI, and permutation test (N=20) per condition.
Results averaged over 5 random seeds.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transfer_learning.multi_source_utils import (
    WEIGHTS_DIR,
    bootstrap_auc_ci,
    load_splits,
    majority_baseline,
    permutation_test,
    prepare_pyg_data,
    print_summary_table,
    run_training,
    save_results,
)


CONDITION_CONFIGS = {
    "MS-A": {
        "pretrained": True,
        "freeze_backbone": True,
        "lr": 1e-3,
        "head_hidden": 64,
        "description": "USSF pretrained + frozen (linear probe)",
    },
    "MS-B": {
        "pretrained": True,
        "freeze_backbone": False,
        "lr": 1e-4,
        "head_hidden": 64,
        "description": "USSF pretrained + fine-tuned",
    },
    "MS-C": {
        "pretrained": False,
        "freeze_backbone": False,
        "lr": 1e-3,
        "head_hidden": 64,
        "description": "Random init (from scratch)",
    },
}


def run_exp1(
    seeds: list,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    n_permutations: int = 20,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Experiment 1: Full Multi-Source Dataset")
    print(f"  {datetime.now()}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print("=" * 70)

    train_raw, val_raw, test_raw = load_splits()
    print(f"  Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    all_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        train_data = prepare_pyg_data(train_raw)
        val_data = prepare_pyg_data(val_raw)
        test_data = prepare_pyg_data(test_raw)

        # MS-D: Majority baseline (no seed dependence, but store per-seed for consistency)
        baseline = majority_baseline(train_data, val_data, test_data)
        baseline_ci = bootstrap_auc_ci(baseline["test_metrics"], seed=seed)
        baseline["bootstrap_ci"] = baseline_ci
        all_results[seed]["MS-D"] = baseline
        print(f"\n  MS-D (majority): AUC={baseline['test_metrics']['auc']:.4f}")

        # Trained conditions
        for cond_name, cfg in CONDITION_CONFIGS.items():
            model, result = run_training(
                train_data,
                val_data,
                test_data,
                pretrained=cfg["pretrained"],
                freeze_backbone=cfg["freeze_backbone"],
                lr=cfg["lr"],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                head_hidden=cfg["head_hidden"],
                device=device,
                seed=seed,
                label=f"{cond_name} (seed={seed})",
            )

            # Bootstrap CI
            ci = bootstrap_auc_ci(result["test_metrics"], seed=seed)
            result["bootstrap_ci"] = ci

            # Permutation test: load backbone state for frozen head training
            backbone_path = WEIGHTS_DIR / "ussf_backbone_dense.pt"
            if backbone_path.exists():
                backbone_state = torch.load(
                    backbone_path, map_location=device, weights_only=False
                )
            else:
                # Use the trained model's backbone
                backbone_state = {
                    "conv1": model.conv1.state_dict(),
                    "lin_in": model.lin_in.state_dict(),
                    "convs": [c.state_dict() for c in model.convs],
                }

            perm = permutation_test(
                backbone_state=backbone_state,
                train_data=train_data,
                test_data=test_data,
                n_permutations=n_permutations,
                head_hidden=cfg["head_hidden"],
                device=device,
                seed=seed,
                observed_auc=result["test_metrics"]["auc"],
            )
            result["permutation_test"] = perm
            print(
                f"  Permutation test: p={perm['p_value']:.3f} "
                f"(null={perm['null_auc_mean']:.3f}±{perm['null_auc_std']:.3f})"
            )

            all_results[seed][cond_name] = result

    # Aggregate across seeds
    conditions = list(CONDITION_CONFIGS.keys()) + ["MS-D"]
    aggregated = {}
    for cond in conditions:
        aucs, accs, f1s = [], [], []
        p_values = []
        for seed in seeds:
            if cond in all_results[seed]:
                r = all_results[seed][cond]
                aucs.append(r["test_metrics"]["auc"])
                accs.append(r["test_metrics"]["accuracy"])
                f1s.append(r["test_metrics"]["f1"])
                if "permutation_test" in r:
                    p_values.append(r["permutation_test"]["p_value"])

        if aucs:
            aggregated[cond] = {
                "test_auc_mean": float(np.mean(aucs)),
                "test_auc_std": float(np.std(aucs)),
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_std": float(np.std(accs)),
                "test_f1_mean": float(np.mean(f1s)),
                "test_f1_std": float(np.std(f1s)),
                "n_seeds": len(aucs),
                "aucs": aucs,
                "accs": accs,
                "f1s": f1s,
                "p_values": p_values,
            }

    # Print summary
    print_summary_table(aggregated, "Experiment 1: Full Multi-Source Dataset")

    # Print permutation test summary
    print("\n  Permutation tests (N={}):".format(n_permutations))
    for cond in conditions:
        if cond in aggregated and aggregated[cond]["p_values"]:
            pvs = aggregated[cond]["p_values"]
            print(f"    {cond}: p-values = {[f'{p:.3f}' for p in pvs]}")

    output = {
        "experiment": "exp1_full_dataset",
        "timestamp": str(datetime.now()),
        "config": {
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "n_permutations": n_permutations,
            "conditions": {k: v["description"] for k, v in CONDITION_CONFIGS.items()},
        },
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    save_results(output, "exp1_full_dataset.pkl")
    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Full Multi-Source Dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_exp1(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        n_permutations=args.n_permutations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
