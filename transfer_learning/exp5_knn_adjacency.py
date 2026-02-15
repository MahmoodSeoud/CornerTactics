#!/usr/bin/env python3
"""
Experiment 5: kNN Adjacency Graph Construction
================================================

Question: "Does dense adjacency drown the signal in noise for corners,
where spatial proximity defines tactical relationships (marking)?"

Replaces dense (fully-connected) graphs with kNN (k=5 nearest neighbors
by Euclidean distance). Retrains CrystalConv on the full 3,078 dataset.

Conditions:
    kNN-A: USSF pretrained + frozen backbone (linear probe)
    kNN-B: USSF pretrained + fine-tuned (unfrozen)
    kNN-C: Random init (train from scratch)
    kNN-D: Majority class baseline

Reports AUC, F1, bootstrap 95% CI, and permutation test (N=20) per condition.
Results averaged over 5 random seeds.
"""

import argparse
import pickle
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
    rebuild_knn_edges,
    run_training,
    save_results,
)


CONDITION_CONFIGS = {
    "kNN-A": {
        "pretrained": True,
        "freeze_backbone": True,
        "lr": 1e-3,
        "head_hidden": 64,
        "description": "USSF pretrained + frozen (linear probe), kNN edges",
    },
    "kNN-B": {
        "pretrained": True,
        "freeze_backbone": False,
        "lr": 1e-4,
        "head_hidden": 64,
        "description": "USSF pretrained + fine-tuned, kNN edges",
    },
    "kNN-C": {
        "pretrained": False,
        "freeze_backbone": False,
        "lr": 1e-3,
        "head_hidden": 64,
        "description": "Random init (from scratch), kNN edges",
    },
}


def run_exp5(
    seeds: list,
    k: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    n_permutations: int = 20,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"  Experiment 5: kNN Adjacency (k={k})")
    print(f"  {datetime.now()}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load dense data then rebuild with kNN
    train_raw, val_raw, test_raw = load_splits()
    print(f"  Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    all_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        # Convert to PyG then rebuild edges
        train_data = rebuild_knn_edges(prepare_pyg_data(train_raw), k=k)
        val_data = rebuild_knn_edges(prepare_pyg_data(val_raw), k=k)
        test_data = rebuild_knn_edges(prepare_pyg_data(test_raw), k=k)

        # Report edge stats
        train_edges = [d.edge_index.shape[1] for d in train_data]
        print(f"  kNN edges: mean={np.mean(train_edges):.1f}, "
              f"min={np.min(train_edges)}, max={np.max(train_edges)}")

        # kNN-D: Majority baseline
        baseline = majority_baseline(train_data, val_data, test_data)
        baseline_ci = bootstrap_auc_ci(baseline["test_metrics"], seed=seed)
        baseline["bootstrap_ci"] = baseline_ci
        all_results[seed]["kNN-D"] = baseline
        print(f"\n  kNN-D (majority): AUC={baseline['test_metrics']['auc']:.4f}")

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

            # Permutation test
            backbone_path = WEIGHTS_DIR / "ussf_backbone_dense.pt"
            if backbone_path.exists():
                backbone_state = torch.load(
                    backbone_path, map_location=device, weights_only=False
                )
            else:
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
    conditions = list(CONDITION_CONFIGS.keys()) + ["kNN-D"]
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

    # Print kNN results
    print_summary_table(aggregated, f"Experiment 5: kNN Adjacency (k={k})")

    # Print comparison with dense baseline (exp1)
    exp1_path = Path(__file__).parent.parent / "results" / "multi_source_experiments" / "exp1_full_dataset.pkl"
    if exp1_path.exists():
        exp1 = pickle.load(open(exp1_path, "rb"))
        dense_agg = exp1.get("aggregated", {})
        print(f"\n{'='*70}")
        print(f"  Comparison: kNN (k={k}) vs Dense")
        print(f"{'='*70}")
        print(f"  {'Condition':<20} {'kNN AUC':>14} {'Dense AUC':>14} {'Delta':>10}")
        print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*10}")
        pairs = [
            ("kNN-A", "MS-A"), ("kNN-B", "MS-B"),
            ("kNN-C", "MS-C"), ("kNN-D", "MS-D"),
        ]
        for knn_c, dense_c in pairs:
            if knn_c in aggregated and dense_c in dense_agg:
                ka = aggregated[knn_c]["test_auc_mean"]
                ks = aggregated[knn_c]["test_auc_std"]
                da = dense_agg[dense_c]["test_auc_mean"]
                ds = dense_agg[dense_c]["test_auc_std"]
                delta = ka - da
                print(f"  {knn_c:<20} {ka:.3f}±{ks:.3f}    {da:.3f}±{ds:.3f}    {delta:+.3f}")
        print(f"{'='*70}")

    # Permutation test summary
    print(f"\n  Permutation tests (N={n_permutations}):")
    for cond in conditions:
        if cond in aggregated and aggregated[cond]["p_values"]:
            pvs = aggregated[cond]["p_values"]
            print(f"    {cond}: p-values = {[f'{p:.3f}' for p in pvs]}")

    output = {
        "experiment": "exp5_knn_adjacency",
        "timestamp": str(datetime.now()),
        "config": {
            "k": k,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "n_permutations": n_permutations,
            "conditions": {c: v["description"] for c, v in CONDITION_CONFIGS.items()},
        },
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    save_results(output, "exp5_knn_adjacency.pkl")
    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: kNN Adjacency")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--k", type=int, default=5, help="kNN neighborhood size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_exp5(
        seeds=args.seeds,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        n_permutations=args.n_permutations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
