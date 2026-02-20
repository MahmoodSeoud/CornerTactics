"""Leave-One-Match-Out (LOMO) cross-validation for corner kick prediction.

Runs 10-fold CV where each fold holds out one match for testing.
Trains both stages and evaluates receiver prediction + shot prediction
under oracle, predicted, and unconditional receiver modes.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from corner_prediction.config import (
    BATCH_SIZE,
    MATCH_IDS,
    RECEIVER_DROPOUT,
    RECEIVER_EPOCHS,
    RECEIVER_HIDDEN,
    RECEIVER_LR,
    RECEIVER_PATIENCE,
    RECEIVER_WEIGHT_DECAY,
    RESULTS_DIR,
    SHOT_DROPOUT,
    SHOT_EPOCHS,
    SHOT_HIDDEN,
    SHOT_LR,
    SHOT_PATIENCE,
    SHOT_POS_WEIGHT,
    SHOT_WEIGHT_DECAY,
)
from corner_prediction.data.dataset import get_match_ids, lomo_split
from corner_prediction.training.train import (
    build_model,
    eval_receiver,
    eval_shot,
    train_fold,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def compute_receiver_metrics(fold_results: List[Dict]) -> Dict:
    """Aggregate receiver metrics across LOMO folds.

    Returns dict with mean, std, and per-fold values for top1/top3 accuracy.
    """
    top1s = [f["receiver"]["top1_acc"] for f in fold_results if f["receiver"]["n_labeled"] > 0]
    top3s = [f["receiver"]["top3_acc"] for f in fold_results if f["receiver"]["n_labeled"] > 0]

    return {
        "top1_mean": float(np.mean(top1s)) if top1s else 0.0,
        "top1_std": float(np.std(top1s)) if top1s else 0.0,
        "top3_mean": float(np.mean(top3s)) if top3s else 0.0,
        "top3_std": float(np.std(top3s)) if top3s else 0.0,
        "n_folds": len(top1s),
        "per_fold_top1": top1s,
        "per_fold_top3": top3s,
    }


def compute_shot_metrics(fold_results: List[Dict], mode: str = "oracle") -> Dict:
    """Aggregate shot metrics across LOMO folds for a given receiver mode.

    Args:
        mode: "oracle" | "predicted" | "unconditional"
    """
    key = f"shot_{mode}"
    aucs = [f[key]["auc"] for f in fold_results]
    f1s = [f[key]["f1"] for f in fold_results]
    accs = [f[key]["accuracy"] for f in fold_results]

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "n_folds": len(aucs),
        "per_fold_auc": aucs,
        "per_fold_f1": f1s,
    }


# ---------------------------------------------------------------------------
# LOMO Cross-Validation
# ---------------------------------------------------------------------------


def lomo_cv(
    dataset,
    backbone_mode: str = "pretrained",
    pretrained_path: Optional[str] = None,
    freeze: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    # Allow overriding hyperparameters for ablations
    receiver_lr: float = RECEIVER_LR,
    receiver_epochs: int = RECEIVER_EPOCHS,
    receiver_patience: int = RECEIVER_PATIENCE,
    receiver_weight_decay: float = RECEIVER_WEIGHT_DECAY,
    shot_lr: float = SHOT_LR,
    shot_epochs: int = SHOT_EPOCHS,
    shot_patience: int = SHOT_PATIENCE,
    shot_weight_decay: float = SHOT_WEIGHT_DECAY,
    shot_pos_weight: float = SHOT_POS_WEIGHT,
    batch_size: int = BATCH_SIZE,
    linear_heads: bool = False,
) -> Dict[str, Any]:
    """Run 10-fold Leave-One-Match-Out cross-validation.

    For each fold:
        1. Hold out one match for testing.
        2. Use the next match (in sorted order) as validation.
        3. Train on the remaining 8 matches.
        4. Evaluate receiver and shot prediction.

    Returns:
        Dict with per-fold results and aggregated metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match_ids = get_match_ids(dataset)
    fold_results = []

    for fold_idx, held_out in enumerate(match_ids):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{len(match_ids)}: held_out={held_out} ---")

        # Split: test = held_out, val = next match, train = rest
        train_data, test_data = lomo_split(dataset, held_out)

        if not test_data:
            logger.warning("No test data for match %s, skipping fold", held_out)
            continue

        # Inner validation: hold out next match from training
        val_match = match_ids[(fold_idx + 1) % len(match_ids)]
        inner_train = [g for g in train_data if str(g.match_id) != val_match]
        val_data = [g for g in train_data if str(g.match_id) == val_match]

        if not val_data:
            # Fallback: use first 20% of train as val
            n_val = max(1, len(train_data) // 5)
            val_data = train_data[:n_val]
            inner_train = train_data[n_val:]

        if verbose:
            n_train_shots = sum(1 for g in inner_train if g.shot_label == 1)
            n_test_shots = sum(1 for g in test_data if g.shot_label == 1)
            print(f"  train={len(inner_train)} ({n_train_shots} shots), "
                  f"val={len(val_data)}, test={len(test_data)} ({n_test_shots} shots)")

        # Build fresh model for each fold
        model = build_model(
            backbone_mode=backbone_mode,
            pretrained_path=pretrained_path,
            freeze=freeze,
            receiver_hidden=RECEIVER_HIDDEN,
            receiver_dropout=RECEIVER_DROPOUT,
            shot_hidden=SHOT_HIDDEN,
            shot_dropout=SHOT_DROPOUT,
            linear_heads=linear_heads,
        ).to(device)

        # Train both stages
        model, loss_history = train_fold(
            model, inner_train, val_data, device,
            receiver_lr=receiver_lr,
            receiver_epochs=receiver_epochs,
            receiver_patience=receiver_patience,
            receiver_weight_decay=receiver_weight_decay,
            shot_lr=shot_lr,
            shot_epochs=shot_epochs,
            shot_patience=shot_patience,
            shot_weight_decay=shot_weight_decay,
            shot_pos_weight=shot_pos_weight,
            batch_size=batch_size,
        )

        # Evaluate
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        receiver_metrics = eval_receiver(model, test_loader, device)
        shot_oracle = eval_shot(model, test_loader, device, receiver_mode="oracle")
        shot_predicted = eval_shot(model, test_loader, device, receiver_mode="predicted")
        shot_unconditional = eval_shot(model, test_loader, device, receiver_mode="none")

        fold_result = {
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(inner_train),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "receiver": receiver_metrics,
            "shot_oracle": shot_oracle,
            "shot_predicted": shot_predicted,
            "shot_unconditional": shot_unconditional,
            "loss_history": loss_history,
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Receiver: top1={receiver_metrics['top1_acc']:.3f}, "
                  f"top3={receiver_metrics['top3_acc']:.3f} "
                  f"(n={receiver_metrics['n_labeled']})")
            print(f"  Shot (oracle):        AUC={shot_oracle['auc']:.3f}, "
                  f"F1={shot_oracle['f1']:.3f}")
            print(f"  Shot (predicted):     AUC={shot_predicted['auc']:.3f}, "
                  f"F1={shot_predicted['f1']:.3f}")
            print(f"  Shot (unconditional): AUC={shot_unconditional['auc']:.3f}, "
                  f"F1={shot_unconditional['f1']:.3f}")

    # Aggregate
    agg_receiver = compute_receiver_metrics(fold_results)
    agg_shot_oracle = compute_shot_metrics(fold_results, "oracle")
    agg_shot_predicted = compute_shot_metrics(fold_results, "predicted")
    agg_shot_unconditional = compute_shot_metrics(fold_results, "unconditional")

    results = {
        "config": {
            "backbone_mode": backbone_mode,
            "freeze": freeze,
            "seed": seed,
            "n_folds": len(fold_results),
            "linear_heads": linear_heads,
        },
        "per_fold": fold_results,
        "aggregated": {
            "receiver": agg_receiver,
            "shot_oracle": agg_shot_oracle,
            "shot_predicted": agg_shot_predicted,
            "shot_unconditional": agg_shot_unconditional,
        },
    }

    if verbose:
        print_results_table(results)

    return results


def print_results_table(results: Dict) -> None:
    """Print thesis-ready results table."""
    agg = results["aggregated"]

    print(f"\n{'=' * 70}")
    print("LOMO Cross-Validation Results")
    print(f"{'=' * 70}")

    # Receiver
    r = agg["receiver"]
    print(f"\nStage 1 — Receiver Prediction ({r['n_folds']} folds):")
    print(f"  Top-1 accuracy: {r['top1_mean']:.3f} +/- {r['top1_std']:.3f}")
    print(f"  Top-3 accuracy: {r['top3_mean']:.3f} +/- {r['top3_std']:.3f}")
    print(f"  Random baseline: top1 ~0.147, top3 ~0.441")

    # Shot
    print(f"\nStage 2 — Shot Prediction:")
    for mode, label in [
        ("shot_oracle", "Oracle receiver"),
        ("shot_predicted", "Predicted receiver"),
        ("shot_unconditional", "Unconditional"),
    ]:
        s = agg[mode]
        print(f"  {label:22s}: AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f}, "
              f"F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f}")
    print(f"  Random baseline:        AUC=0.500")
    print(f"{'=' * 70}")


def save_results(results: Dict, name: str = "lomo", output_dir: Optional[str] = None) -> Path:
    """Save results as pickle + JSON summary.

    Handles different result formats (LOMO, permutation, ablation).
    """
    out = Path(output_dir) if output_dir else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results (pickle)
    pkl_path = out / f"{name}_{timestamp}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    # JSON summary — extract serializable top-level keys, skip large arrays
    summary = {"timestamp": timestamp}
    for k, v in results.items():
        if k == "per_fold":
            summary["n_folds"] = len(v)
        elif k == "null_distribution":
            summary["n_permutations"] = len(v)
        elif isinstance(v, (str, int, float, bool, dict)):
            summary[k] = v
        elif isinstance(v, list) and len(v) < 50:
            summary[k] = v

    json_path = out / f"{name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to %s", pkl_path)
    return pkl_path
