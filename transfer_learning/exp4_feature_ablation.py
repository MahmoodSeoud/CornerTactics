#!/usr/bin/env python3
"""
Experiment 4: Feature Ablation (Velocity vs Position)
======================================================

Question: "Do velocity vectors provide the missing signal, now that we
have enough data?"

Approach: Train two separate models:
    F-A: All 12 features (full model)
    F-B: Position only (velocity features zeroed in both nodes and edges)

Compare test AUC between the two, with bootstrap CI and permutation test.

Velocity features zeroed:
    Node: [2] vx, [3] vy, [4] velocity_mag, [5] velocity_angle
    Edge: [1] speed_diff, [4] vel_sin, [5] vel_cos
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
    permutation_test,
    prepare_pyg_data,
    run_training,
    save_results,
    zero_velocity_features,
)


def run_exp4(
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
    print("  Experiment 4: Feature Ablation (Velocity vs Position)")
    print(f"  {datetime.now()}")
    print("=" * 70)

    train_raw, val_raw, test_raw = load_splits()

    all_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        # Full features
        train_data = prepare_pyg_data(train_raw)
        val_data = prepare_pyg_data(val_raw)
        test_data = prepare_pyg_data(test_raw)

        # Position only (velocity zeroed)
        train_pos = zero_velocity_features(train_data)
        val_pos = zero_velocity_features(val_data)
        test_pos = zero_velocity_features(test_data)

        # F-A: Full features
        model_a, result_a = run_training(
            train_data,
            val_data,
            test_data,
            pretrained=True,
            freeze_backbone=False,
            lr=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
            label=f"F-A Full features (seed={seed})",
        )
        result_a["bootstrap_ci"] = bootstrap_auc_ci(
            result_a["test_metrics"], seed=seed
        )

        # F-B: Position only
        model_b, result_b = run_training(
            train_pos,
            val_pos,
            test_pos,
            pretrained=True,
            freeze_backbone=False,
            lr=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
            label=f"F-B Position only (seed={seed})",
        )
        result_b["bootstrap_ci"] = bootstrap_auc_ci(
            result_b["test_metrics"], seed=seed
        )

        # Permutation tests for both
        backbone_path = WEIGHTS_DIR / "ussf_backbone_dense.pt"
        if backbone_path.exists():
            backbone_state = torch.load(
                backbone_path, map_location=device, weights_only=False
            )
        else:
            backbone_state = {
                "conv1": model_a.conv1.state_dict(),
                "lin_in": model_a.lin_in.state_dict(),
                "convs": [c.state_dict() for c in model_a.convs],
            }

        perm_a = permutation_test(
            backbone_state=backbone_state,
            train_data=train_data,
            test_data=test_data,
            n_permutations=n_permutations,
            device=device,
            seed=seed,
            observed_auc=result_a["test_metrics"]["auc"],
        )
        result_a["permutation_test"] = perm_a
        print(
            f"  F-A perm test: p={perm_a['p_value']:.3f} "
            f"(null={perm_a['null_auc_mean']:.3f}±{perm_a['null_auc_std']:.3f})"
        )

        perm_b = permutation_test(
            backbone_state=backbone_state,
            train_data=train_pos,
            test_data=test_pos,
            n_permutations=n_permutations,
            device=device,
            seed=seed,
            observed_auc=result_b["test_metrics"]["auc"],
        )
        result_b["permutation_test"] = perm_b
        print(
            f"  F-B perm test: p={perm_b['p_value']:.3f} "
            f"(null={perm_b['null_auc_mean']:.3f}±{perm_b['null_auc_std']:.3f})"
        )

        all_results[seed] = {
            "F-A": result_a,
            "F-B": result_b,
        }

    # Aggregate
    aggregated = {}
    for cond in ["F-A", "F-B"]:
        aucs = [all_results[s][cond]["test_metrics"]["auc"] for s in seeds]
        f1s = [all_results[s][cond]["test_metrics"]["f1"] for s in seeds]
        accs = [all_results[s][cond]["test_metrics"]["accuracy"] for s in seeds]
        p_vals = [all_results[s][cond]["permutation_test"]["p_value"] for s in seeds]
        aggregated[cond] = {
            "test_auc_mean": float(np.mean(aucs)),
            "test_auc_std": float(np.std(aucs)),
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s)),
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "aucs": aucs,
            "p_values": p_vals,
        }

    # Paired comparison
    auc_diffs = [
        all_results[s]["F-A"]["test_metrics"]["auc"]
        - all_results[s]["F-B"]["test_metrics"]["auc"]
        for s in seeds
    ]

    # Print summary
    print(f"\n{'='*70}")
    print("  Experiment 4: Feature Ablation Results")
    print(f"{'='*70}")
    print(f"  {'Condition':<25} {'AUC':>14} {'F1':>14} {'p-value':>10}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")
    for cond, desc in [("F-A", "All features"), ("F-B", "Position only")]:
        a = aggregated[cond]
        avg_p = float(np.mean(a["p_values"]))
        print(
            f"  {desc:<25} "
            f"{a['test_auc_mean']:.3f}±{a['test_auc_std']:.3f}   "
            f"{a['test_f1_mean']:.3f}±{a['test_f1_std']:.3f}   "
            f"{avg_p:.3f}"
        )
    print(f"\n  AUC difference (F-A - F-B): {np.mean(auc_diffs):.4f} ± {np.std(auc_diffs):.4f}")
    print(f"  Per-seed diffs: {[f'{d:.4f}' for d in auc_diffs]}")
    print(f"{'='*70}")

    output = {
        "experiment": "exp4_feature_ablation",
        "timestamp": str(datetime.now()),
        "config": {
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_permutations": n_permutations,
            "velocity_node_features_zeroed": [2, 3, 4, 5],
            "velocity_edge_features_zeroed": [1, 4, 5],
        },
        "per_seed": all_results,
        "aggregated": aggregated,
        "paired_auc_diff": {
            "mean": float(np.mean(auc_diffs)),
            "std": float(np.std(auc_diffs)),
            "per_seed": auc_diffs,
        },
    }

    save_results(output, "exp4_feature_ablation.pkl")
    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Feature Ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_exp4(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        n_permutations=args.n_permutations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
