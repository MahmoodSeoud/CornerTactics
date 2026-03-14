#!/usr/bin/env python3
"""MLP baseline on StatsBomb freeze-frame data with grouped k-fold CV.

Flattens 23 nodes x 12 USSF features = 276-dim vector, then trains the same
ShotMLP architecture used in the tracking-data baseline. StatsBomb freeze-frames
have no velocity, so features 2-5 are already zero from build_dataset().

Usage:
    cd /home/mseo/CornerTactics
    source FAANTRA/venv/bin/activate
    python scripts/statsbomb_mlp_baseline.py                    # multi-seed eval
    python scripts/statsbomb_mlp_baseline.py --permutation      # + permutation test
    python scripts/statsbomb_mlp_baseline.py --eval-only        # single seed eval
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corner_prediction.config import (
    BATCH_SIZE,
    N_PERMUTATIONS,
    RESULTS_DIR,
    SEEDS,
    SHOT_DROPOUT,
    SHOT_EPOCHS,
    SHOT_LR,
    SHOT_PATIENCE,
    SHOT_POS_WEIGHT,
    SHOT_WEIGHT_DECAY,
)
from corner_prediction.baselines.mlp_baseline import (
    ShotMLP,
    _build_tensors,
)
from corner_prediction.training.permutation_test import shuffle_shot_labels
from scripts.statsbomb_ussf_pipeline import build_dataset

# Fixed dimensions for USSF-aligned StatsBomb graphs
EXPECTED_NODES = 23      # 22 players + 1 ball (padded if fewer)
EXPECTED_FEATURES = 12   # USSF 12-feature schema
INPUT_DIM = EXPECTED_NODES * EXPECTED_FEATURES  # 276


def _mlp_fold_fixed(
    train_data: list,
    val_data: list,
    test_data: list,
    seed: int,
    device: torch.device,
    hidden_dim: int = 64,
    dropout: float = SHOT_DROPOUT,
    lr: float = SHOT_LR,
    epochs: int = SHOT_EPOCHS,
    patience: int = SHOT_PATIENCE,
    weight_decay: float = SHOT_WEIGHT_DECAY,
    pos_weight: float = SHOT_POS_WEIGHT,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train and evaluate MLP on one fold with fixed 23x12=276 input dim.

    Unlike _mlp_fold which auto-detects dims from the first graph (unreliable
    when StatsBomb graphs have variable player counts), this always pads/truncates
    to EXPECTED_NODES x EXPECTED_FEATURES.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = _build_tensors(train_data, EXPECTED_NODES, EXPECTED_FEATURES)
    X_val, y_val = _build_tensors(val_data, EXPECTED_NODES, EXPECTED_FEATURES)
    X_test, y_test = _build_tensors(test_data, EXPECTED_NODES, EXPECTED_FEATURES)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ShotMLP(input_dim=INPUT_DIM, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor([pos_weight], device=device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pw)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, y_val.to(device), pos_weight=pw,
            ).item()

        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze(-1)
        probs = torch.sigmoid(test_logits).cpu().numpy()

    y_test_np = y_test.numpy()

    # AUC
    if len(np.unique(y_test_np)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_test_np, probs)

    # F1 at optimal threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_test_np, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_05 = (probs >= 0.5).astype(int)
    accuracy = float((preds_05 == y_test_np).mean())

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "accuracy": accuracy,
        "probs": probs.tolist(),
        "labels": y_test_np.tolist(),
        "n_samples": len(y_test_np),
        "n_positive": int(y_test_np.sum()),
        "loss_history": {"train": train_losses, "val": val_losses},
    }


def grouped_kfold_mlp(
    dataset: list,
    n_folds: int = 10,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """Grouped k-fold CV for MLP baseline on StatsBomb data.

    Groups by match_id so no match leaks across train/val/test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Group matches into k folds
    match_ids = sorted(set(str(g.match_id) for g in dataset))
    rng = np.random.RandomState(seed)
    rng.shuffle(match_ids)
    fold_matches = [match_ids[i::n_folds] for i in range(n_folds)]

    fold_results = []

    for fold_idx in range(n_folds):
        test_matches = set(fold_matches[fold_idx])
        val_matches = set(fold_matches[(fold_idx + 1) % n_folds])

        test_data = [g for g in dataset if str(g.match_id) in test_matches]
        val_data = [g for g in dataset if str(g.match_id) in val_matches]
        train_data = [g for g in dataset
                      if str(g.match_id) not in test_matches
                      and str(g.match_id) not in val_matches]

        # Skip if test set lacks class diversity
        test_labels = [g.shot_label for g in test_data]
        if not test_data or len(set(test_labels)) < 2:
            if verbose:
                reason = "empty" if not test_data else "single class"
                print(f"  Fold {fold_idx+1}/{n_folds}: skipped ({reason})")
            continue

        if verbose:
            n_train_shots = sum(1 for g in train_data if g.shot_label == 1)
            n_test_shots = sum(1 for g in test_data if g.shot_label == 1)
            print(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_data)} "
                  f"({n_train_shots} shots), val={len(val_data)}, "
                  f"test={len(test_data)} ({n_test_shots} shots)")

        fold_seed = seed + fold_idx
        shot = _mlp_fold_fixed(train_data, val_data, test_data, fold_seed, device)

        fold_results.append({
            "fold_idx": fold_idx,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "auc": shot["auc"],
            "f1": shot["f1"],
            "accuracy": shot["accuracy"],
        })

        if verbose:
            print(f"    AUC={shot['auc']:.3f}, F1={shot['f1']:.3f}")

    # Aggregate
    aucs = [f["auc"] for f in fold_results]
    f1s = [f["f1"] for f in fold_results]

    auc_mean = float(np.mean(aucs)) if aucs else 0.0
    auc_std = float(np.std(aucs)) if aucs else 0.0
    f1_mean = float(np.mean(f1s)) if f1s else 0.0
    f1_std = float(np.std(f1s)) if f1s else 0.0

    return {
        "fold_results": fold_results,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "n_folds_used": len(fold_results),
        "n_folds_total": n_folds,
        "n_corners": len(dataset),
        "n_matches": len(match_ids),
        "seed": seed,
        "cv_method": f"grouped_{n_folds}fold",
    }


def multi_seed_eval(
    dataset: list,
    n_folds: int = 10,
    seeds: list = None,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """Run grouped k-fold CV across multiple seeds, report mean +/- std."""
    if seeds is None:
        seeds = SEEDS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_results = []
    for seed in seeds:
        if verbose:
            print(f"\n{'─' * 40}")
            print(f"Seed {seed}")
            print(f"{'─' * 40}")
        result = grouped_kfold_mlp(dataset, n_folds=n_folds, seed=seed,
                                   device=device, verbose=verbose)
        seed_results.append(result)
        if verbose:
            print(f"  → AUC={result['auc_mean']:.3f}")

    aucs = [r["auc_mean"] for r in seed_results]
    f1s = [r["f1_mean"] for r in seed_results]

    return {
        "per_seed": seed_results,
        "seeds": seeds,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "per_seed_auc": {s: a for s, a in zip(seeds, aucs)},
    }


def permutation_test_mlp(
    dataset: list,
    n_folds: int = 10,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """Permutation test for MLP baseline using grouped k-fold CV."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: MLP Shot AUC (grouped {n_folds}-fold, "
              f"n={n_permutations})")
        print(f"{'=' * 60}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_result = grouped_kfold_mlp(dataset, n_folds=n_folds, seed=seed,
                                    device=device, verbose=verbose)
    real_auc = real_result["auc_mean"]

    if verbose:
        print(f"\nReal AUC: {real_auc:.4f}")

    # Null distribution
    null_aucs = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_result = grouped_kfold_mlp(shuffled, n_folds=n_folds, seed=seed,
                                        device=device, verbose=False)
        null_aucs.append(perm_result["auc_mean"])

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= real_auc) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_aucs.mean():.4f}, "
              f"std={null_aucs.std():.4f}")
        print(f"p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return {
        "metric": "shot_auc_mlp",
        "real_metric": float(real_auc),
        "null_distribution": null_aucs.tolist(),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": bool(p_value < 0.05),
        "cv_method": f"grouped_{n_folds}fold",
        "n_folds": n_folds,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MLP baseline on StatsBomb freeze-frame data",
    )
    parser.add_argument("--permutation", action="store_true",
                        help="Run permutation test (N=100, seed 42 only)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Single seed evaluation only (no multi-seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of folds for grouped k-fold CV")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("MLP Baseline on StatsBomb Freeze-Frames")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Architecture: ShotMLP({INPUT_DIM}, 64, dropout=0.3)")
    print(f"CV: grouped {args.n_folds}-fold (match-grouped, no leakage)")

    # Build dataset
    graphs = build_dataset()

    n_corners = len(graphs)
    n_matches = len(set(str(g.match_id) for g in graphs))
    n_shots = sum(1 for g in graphs if g.shot_label == 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Multi-seed or single-seed evaluation
    if args.eval_only:
        print(f"\n{'=' * 60}")
        print(f"Single-Seed Evaluation (seed={args.seed})")
        print(f"{'=' * 60}")
        result = grouped_kfold_mlp(graphs, n_folds=args.n_folds, seed=args.seed,
                                   device=device, verbose=True)
        seed42_auc = result["auc_mean"]
        multi_auc_mean = result["auc_mean"]
        multi_auc_std = result["auc_std"]
        multi_f1_mean = result["f1_mean"]
        multi_f1_std = result["f1_std"]
        multi_result = {"per_seed": [result], "seeds": [args.seed]}
    else:
        print(f"\n{'=' * 60}")
        print(f"Multi-Seed Evaluation ({len(SEEDS)} seeds)")
        print(f"{'=' * 60}")
        multi_result = multi_seed_eval(graphs, n_folds=args.n_folds,
                                       device=device, verbose=True)
        multi_auc_mean = multi_result["auc_mean"]
        multi_auc_std = multi_result["auc_std"]
        multi_f1_mean = multi_result["f1_mean"]
        multi_f1_std = multi_result["f1_std"]
        seed42_auc = multi_result["per_seed_auc"].get(42, multi_auc_mean)

    # Save eval results
    eval_out = {
        "model": "mlp",
        "dataset": "statsbomb",
        "input_dim": INPUT_DIM,
        "n_corners": n_corners,
        "n_matches": n_matches,
        "n_shots": n_shots,
        "n_folds": args.n_folds,
        "auc_mean": multi_auc_mean,
        "auc_std": multi_auc_std,
        "f1_mean": multi_f1_mean,
        "f1_std": multi_f1_std,
        "per_seed": multi_result.get("per_seed_auc", {}),
        "timestamp": str(datetime.now()),
    }
    eval_path = RESULTS_DIR / "statsbomb_mlp_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"\nSaved: {eval_path}")

    # Permutation test
    perm_result = None
    if args.permutation:
        perm_result = permutation_test_mlp(
            graphs, n_folds=args.n_folds,
            n_permutations=args.n_permutations,
            seed=42, device=device, verbose=True,
        )

        perm_result["dataset"] = "statsbomb"
        perm_result["model"] = "mlp"
        perm_result["n_corners"] = n_corners
        perm_result["n_matches"] = n_matches
        perm_result["timestamp"] = str(datetime.now())

        json_path = RESULTS_DIR / "statsbomb_mlp_perm_shot.json"
        with open(json_path, "w") as f:
            json.dump(perm_result, f, indent=2)
        print(f"\nSaved: {json_path}")

    # Print report
    print(f"\n{'=' * 60}")
    print("StatsBomb MLP Freeze-Frame Results")
    print(f"{'=' * 60}")
    print(f"Corners: {n_corners} (after preprocessing)")
    print(f"Matches: {n_matches}")
    print(f"Folds: {args.n_folds}")
    print()

    if args.eval_only:
        print(f"Single-seed AUC (seed {args.seed}): {seed42_auc:.3f}")
    else:
        print(f"Multi-seed AUC ({len(SEEDS)} seeds): "
              f"{multi_auc_mean:.3f} +/- {multi_auc_std:.3f}")
        print(f"Multi-seed F1  ({len(SEEDS)} seeds): "
              f"{multi_f1_mean:.3f} +/- {multi_f1_std:.3f}")
        print(f"Seed 42 AUC: {seed42_auc:.3f}")

    if perm_result:
        print()
        print("Permutation test:")
        print(f"  Real AUC (seed 42): {perm_result['real_metric']:.3f}")
        print(f"  Null: {perm_result['null_mean']:.3f} +/- "
              f"{perm_result['null_std']:.3f}")
        print(f"  p-value: {perm_result['p_value']:.3f}")

    # Comparison table
    perm_p = f"{perm_result['p_value']:.3f}" if perm_result else "—"
    print(f"\n{'=' * 60}")
    print("| MLP condition | Data | Features | AUC | p |")
    print("|---|---|---|---|---|")
    print(f"| Tracking (reference) | 143 SK+DFL | 276 USSF (vel=0) | 0.602 | 0.040 |")
    print(f"| Freeze-frame (this) | {n_corners} SB corners | 276 USSF (vel=0) "
          f"| {multi_auc_mean:.3f} | {perm_p} |")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
