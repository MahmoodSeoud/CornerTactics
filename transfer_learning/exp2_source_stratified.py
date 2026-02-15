#!/usr/bin/env python3
"""
Experiment 2: Source-Stratified Analysis
=========================================

Question: "Does data quality matter more than quantity?"

Trains on the full multi-source dataset, then evaluates separately on each
data source's test corners.

Known limitation:
    Test split has 0 DFL, 8 SkillCorner, 451 GSR corners.
    DFL results are unavailable; SkillCorner results are extremely noisy.
"""

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader

from transfer_learning.multi_source_utils import (
    bootstrap_auc_ci,
    evaluate,
    load_splits,
    prepare_pyg_data,
    run_training,
    save_results,
)


def run_exp2(
    seeds: list,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Experiment 2: Source-Stratified Analysis")
    print(f"  {datetime.now()}")
    print("=" * 70)

    train_raw, val_raw, test_raw = load_splits()

    # Report source distribution in test set
    test_sources = Counter(s.get("source", "unknown") for s in test_raw)
    print(f"  Test set sources: {dict(test_sources)}")

    all_results = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")

        train_data = prepare_pyg_data(train_raw)
        val_data = prepare_pyg_data(val_raw)
        test_data = prepare_pyg_data(test_raw)

        # Train on full dataset (MS-B config: pretrained + fine-tuned)
        model, result = run_training(
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
            label=f"Full model (seed={seed})",
        )

        seed_result = {
            "full_test": result["test_metrics"],
            "full_test_ci": bootstrap_auc_ci(result["test_metrics"], seed=seed),
            "per_source": {},
        }

        # Evaluate per source
        device_obj = torch.device(device)
        for source_name in ["soccernet_gsr", "skillcorner", "dfl"]:
            subset = [d for d in test_data if d.source == source_name]
            n_subset = len(subset)
            n_pos = sum(1 for d in subset if d.y.item() == 1.0)
            unique_labels = set(d.y.item() for d in subset)

            if n_subset < 2 or len(unique_labels) < 2:
                print(
                    f"  {source_name}: {n_subset} samples "
                    f"({n_pos} pos) — SKIPPED (insufficient data)"
                )
                seed_result["per_source"][source_name] = {
                    "n_samples": n_subset,
                    "n_positive": n_pos,
                    "auc": float("nan"),
                    "note": "insufficient data for AUC",
                }
                continue

            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            metrics = evaluate(model, loader, device_obj)
            ci = bootstrap_auc_ci(metrics, seed=seed)

            print(
                f"  {source_name}: n={n_subset}, pos={n_pos}, "
                f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}"
            )

            seed_result["per_source"][source_name] = {
                **metrics,
                "bootstrap_ci": ci,
            }

        all_results[seed] = seed_result

    # Aggregate
    aggregated = {"full": {}, "per_source": {}}

    full_aucs = [all_results[s]["full_test"]["auc"] for s in seeds]
    full_f1s = [all_results[s]["full_test"]["f1"] for s in seeds]
    aggregated["full"] = {
        "test_auc_mean": float(np.mean(full_aucs)),
        "test_auc_std": float(np.std(full_aucs)),
        "test_f1_mean": float(np.mean(full_f1s)),
        "test_f1_std": float(np.std(full_f1s)),
        "aucs": full_aucs,
    }

    for source in ["soccernet_gsr", "skillcorner", "dfl"]:
        src_aucs = []
        src_f1s = []
        for s in seeds:
            src_data = all_results[s]["per_source"].get(source, {})
            auc = src_data.get("auc", float("nan"))
            if not np.isnan(auc):
                src_aucs.append(auc)
                src_f1s.append(src_data.get("f1", 0))

        if src_aucs:
            aggregated["per_source"][source] = {
                "test_auc_mean": float(np.mean(src_aucs)),
                "test_auc_std": float(np.std(src_aucs)),
                "test_f1_mean": float(np.mean(src_f1s)),
                "test_f1_std": float(np.std(src_f1s)),
                "n_seeds_valid": len(src_aucs),
                "aucs": src_aucs,
            }
        else:
            aggregated["per_source"][source] = {
                "note": "no valid AUC across any seed",
            }

    # Print summary
    print(f"\n{'='*70}")
    print("  Experiment 2: Source-Stratified Results")
    print(f"{'='*70}")
    print(f"  {'Source':<20} {'N':>6} {'AUC':>16} {'F1':>16}")
    print(f"  {'-'*20} {'-'*6} {'-'*16} {'-'*16}")

    full = aggregated["full"]
    print(
        f"  {'ALL (full test)':<20} {len(test_raw):>6} "
        f"{full['test_auc_mean']:.3f}±{full['test_auc_std']:.3f}  "
        f"{full['test_f1_mean']:.3f}±{full['test_f1_std']:.3f}"
    )
    for source in ["soccernet_gsr", "skillcorner", "dfl"]:
        n = test_sources.get(source, 0)
        src = aggregated["per_source"].get(source, {})
        if "test_auc_mean" in src:
            print(
                f"  {source:<20} {n:>6} "
                f"{src['test_auc_mean']:.3f}±{src['test_auc_std']:.3f}  "
                f"{src['test_f1_mean']:.3f}±{src['test_f1_std']:.3f}"
            )
        else:
            print(f"  {source:<20} {n:>6} {'N/A':>16} {'N/A':>16}")
    print(f"{'='*70}")

    output = {
        "experiment": "exp2_source_stratified",
        "timestamp": str(datetime.now()),
        "test_source_distribution": dict(test_sources),
        "per_seed": all_results,
        "aggregated": aggregated,
    }

    save_results(output, "exp2_source_stratified.pkl")
    return output


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Source-Stratified Analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_exp2(
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
