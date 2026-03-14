#!/usr/bin/env python3
"""MLP position-only baseline on tracking data (velocity features zeroed).

Loads combined tracking dataset (143 corners, USSF-aligned 23x12), zeros out
velocity features (indices 2-5), then runs the same MLP architecture as the
full-velocity tracking MLP baseline.

Input is still 276-dimensional (23*12), but 4 features per node are zeros.
This gives a clean comparison between tracking-with-velocity and tracking-
without-velocity for the MLP, paralleling the XGBoost position-only ablation.

Usage:
    cd /home/mseo/CornerTactics
    source FAANTRA/venv/bin/activate

    # Multi-seed evaluation + permutation test (default)
    python scripts/tracking_mlp_posonly.py

    # Eval only (no permutation)
    python scripts/tracking_mlp_posonly.py --no-permutation

    # Single seed
    python scripts/tracking_mlp_posonly.py --eval-only
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
from corner_prediction.baselines.mlp_baseline import ShotMLP
from corner_prediction.data.dataset import (
    CornerKickDataset,
    get_match_ids,
    lomo_split,
)
from corner_prediction.training.permutation_test import shuffle_shot_labels

# USSF-aligned: 23 nodes x 12 features = 276
EXPECTED_NODES = 23
EXPECTED_FEATURES = 12
INPUT_DIM = EXPECTED_NODES * EXPECTED_FEATURES  # 276

# Velocity feature indices to zero out (per node)
VELOCITY_INDICES = [2, 3, 4, 5]  # vx_unit, vy_unit, vel_mag, vel_angle


def _zero_velocity(graph):
    """Return a shallow copy of the graph with velocity features zeroed."""
    g = copy.copy(graph)
    x = graph.x.clone()
    x[:, VELOCITY_INDICES] = 0.0
    g.x = x
    return g


def _flatten_graph(graph) -> np.ndarray:
    """Flatten graph node features to a fixed-size vector."""
    x = graph.x.numpy()
    n_nodes, n_feat = x.shape
    if n_nodes < EXPECTED_NODES:
        pad = np.zeros((EXPECTED_NODES - n_nodes, n_feat))
        x = np.vstack([x, pad])
    elif n_nodes > EXPECTED_NODES:
        x = x[:EXPECTED_NODES]
    return x.flatten()


def _build_tensors(data_list):
    """Build flat feature and label tensors from a list of graphs."""
    X = np.array([_flatten_graph(g) for g in data_list], dtype=np.float32)
    y = np.array([g.shot_label for g in data_list], dtype=np.float32).ravel()
    return torch.from_numpy(X), torch.from_numpy(y)


def _mlp_fold(
    train_data, val_data, test_data, seed, device,
    hidden_dim=64, dropout=SHOT_DROPOUT, lr=SHOT_LR,
    epochs=SHOT_EPOCHS, patience=SHOT_PATIENCE,
    weight_decay=SHOT_WEIGHT_DECAY, pos_weight=SHOT_POS_WEIGHT,
    batch_size=BATCH_SIZE,
):
    """Train and evaluate MLP on one LOMO fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = _build_tensors(train_data)
    X_val, y_val = _build_tensors(val_data)
    X_test, y_test = _build_tensors(test_data)

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

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze(-1)
        probs = torch.sigmoid(test_logits).cpu().numpy()

    y_test_np = y_test.numpy()

    if len(np.unique(y_test_np)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_test_np, probs)

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_test_np, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return {
        "auc": float(auc),
        "f1": float(best_f1),
        "f1_threshold": float(best_thresh),
        "n_samples": len(y_test_np),
        "n_positive": int(y_test_np.sum()),
    }


def lomo_mlp_posonly(dataset, seed=42, device=None, verbose=True):
    """LOMO CV for MLP position-only on tracking data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Zero out velocity features
    dataset_zeroed = [_zero_velocity(g) for g in dataset]

    match_ids = get_match_ids(dataset_zeroed)
    fold_aucs = []
    fold_f1s = []
    fold_details = []

    for fold_idx, held_out in enumerate(match_ids):
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        train_data, test_data = lomo_split(dataset_zeroed, held_out)
        if not test_data:
            continue

        # Inner validation split (same as existing MLP baseline)
        val_match = match_ids[(fold_idx + 1) % len(match_ids)]
        inner_train = [g for g in train_data if str(g.match_id) != val_match]
        val_data = [g for g in train_data if str(g.match_id) == val_match]

        if not val_data:
            n_val = max(1, len(train_data) // 5)
            val_data = train_data[:n_val]
            inner_train = train_data[n_val:]

        result = _mlp_fold(inner_train, val_data, test_data, seed + fold_idx, device)
        fold_aucs.append(result["auc"])
        fold_f1s.append(result["f1"])

        fold_details.append({
            "fold_idx": fold_idx,
            "held_out_match": held_out,
            "n_train": len(inner_train),
            "n_val": len(val_data),
            "n_test": len(test_data),
            **result,
        })

        if verbose:
            n_train_shots = sum(1 for g in inner_train if g.shot_label == 1)
            print(f"  Fold {fold_idx+1}/{len(match_ids)}: held_out={held_out}, "
                  f"train={len(inner_train)} ({n_train_shots} shots), "
                  f"test={len(test_data)}, AUC={result['auc']:.3f}, "
                  f"F1={result['f1']:.3f}")

    auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    auc_std = float(np.std(fold_aucs)) if fold_aucs else 0.0
    f1_mean = float(np.mean(fold_f1s)) if fold_f1s else 0.0
    f1_std = float(np.std(fold_f1s)) if fold_f1s else 0.0

    return {
        "config": {
            "baseline": "mlp_position_only",
            "seed": seed,
            "n_folds": len(match_ids),
            "input_dim": INPUT_DIM,
            "velocity_zeroed": VELOCITY_INDICES,
            "cv_method": "lomo",
        },
        "per_fold": fold_details,
        "aggregated": {
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "n_valid_folds": len(fold_aucs),
            "per_fold_aucs": fold_aucs,
            "per_fold_f1s": fold_f1s,
        },
        "n_corners": len(dataset),
        "n_matches": len(match_ids),
    }


def multi_seed_eval(dataset, seeds=None, device=None, verbose=True):
    """Run LOMO CV across multiple seeds."""
    if seeds is None:
        seeds = SEEDS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_aucs = []
    per_seed_f1s = []
    per_seed_results = {}

    for seed in seeds:
        if verbose:
            print(f"\n--- Seed {seed} ---")
        result = lomo_mlp_posonly(dataset, seed=seed, device=device, verbose=verbose)
        auc = result["aggregated"]["auc_mean"]
        f1 = result["aggregated"]["f1_mean"]
        per_seed_aucs.append(auc)
        per_seed_f1s.append(f1)
        per_seed_results[seed] = result
        if verbose:
            print(f"  Seed {seed}: AUC={auc:.3f}, F1={f1:.3f}")

    return {
        "per_seed": per_seed_results,
        "multi_seed": {
            "auc_mean": float(np.mean(per_seed_aucs)),
            "auc_std": float(np.std(per_seed_aucs)),
            "f1_mean": float(np.mean(per_seed_f1s)),
            "f1_std": float(np.std(per_seed_f1s)),
            "per_seed_aucs": {s: a for s, a in zip(seeds, per_seed_aucs)},
            "per_seed_f1s": {s: f for s, f in zip(seeds, per_seed_f1s)},
            "seeds": seeds,
        },
    }


def permutation_test_mlp_posonly(dataset, n_permutations=N_PERMUTATIONS,
                                  seed=42, device=None, verbose=True):
    """Permutation test for MLP position-only on tracking data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Permutation Test: MLP Position-Only (LOMO, n={n_permutations})")
        print(f"{'=' * 60}")

    # Real metric
    if verbose:
        print("Computing real metric...")
    real_result = lomo_mlp_posonly(dataset, seed=seed, device=device, verbose=verbose)
    real_auc = real_result["aggregated"]["auc_mean"]

    if verbose:
        print(f"\nReal AUC: {real_auc:.4f}")

    # Null distribution
    null_aucs = []
    rng = np.random.RandomState(seed)

    for i in range(n_permutations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        shuffled = shuffle_shot_labels(dataset, rng)
        perm_result = lomo_mlp_posonly(shuffled, seed=seed, device=device, verbose=False)
        null_aucs.append(perm_result["aggregated"]["auc_mean"])

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= real_auc) + 1) / (n_permutations + 1)

    if verbose:
        print(f"\nNull distribution: mean={null_aucs.mean():.4f}, "
              f"std={null_aucs.std():.4f}")
        print(f"p-value: {p_value:.4f} "
              f"{'***' if p_value < 0.01 else '**' if p_value < 0.05 else '(not significant)'}")

    return {
        "metric": "shot_auc_mlp_posonly",
        "real_metric": float(real_auc),
        "null_distribution": null_aucs.tolist(),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": bool(p_value < 0.05),
        "cv_method": "lomo",
    }


def load_tracking_dataset():
    """Load combined tracking data (143 corners) as USSF-aligned graphs."""
    data_dir = Path(__file__).resolve().parent.parent / "corner_prediction" / "data"
    ds = CornerKickDataset(
        root=str(data_dir),
        records_file="combined_corners.pkl",
        edge_type="dense",
        k=6,
        feature_mode="ussf_aligned",
    )
    return list(ds)


def main():
    parser = argparse.ArgumentParser(
        description="MLP position-only baseline on tracking data (velocity zeroed)",
    )
    parser.add_argument("--no-permutation", action="store_true",
                        help="Skip permutation test")
    parser.add_argument("--eval-only", action="store_true",
                        help="Single seed evaluation only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"{'=' * 60}")
    print("MLP Position-Only Baseline on Tracking Data")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Architecture: ShotMLP({INPUT_DIM}, 64, dropout=0.3)")
    print(f"Velocity features zeroed: indices {VELOCITY_INDICES}")
    print(f"CV: LOMO (17 folds)")

    dataset = load_tracking_dataset()
    n_corners = len(dataset)
    n_matches = len(get_match_ids(dataset))
    n_shots = sum(1 for g in dataset if g.shot_label == 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Corners: {n_corners}, Matches: {n_matches}, Shots: {n_shots}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Multi-seed or single-seed evaluation
    if args.eval_only:
        print(f"\n--- Single-Seed Evaluation (seed={args.seed}) ---")
        result = lomo_mlp_posonly(dataset, seed=args.seed, device=device, verbose=True)
        seed42_auc = result["aggregated"]["auc_mean"]
        multi_auc_mean = seed42_auc
        multi_auc_std = result["aggregated"]["auc_std"]
        multi_f1_mean = result["aggregated"]["f1_mean"]
        multi_f1_std = result["aggregated"]["f1_std"]
        multi_result = {"multi_seed": {
            "auc_mean": multi_auc_mean, "auc_std": multi_auc_std,
            "f1_mean": multi_f1_mean, "f1_std": multi_f1_std,
            "per_seed_aucs": {args.seed: seed42_auc},
        }}
    else:
        print(f"\n--- Multi-Seed Evaluation ({len(SEEDS)} seeds) ---")
        multi_result = multi_seed_eval(dataset, device=device, verbose=True)
        multi_auc_mean = multi_result["multi_seed"]["auc_mean"]
        multi_auc_std = multi_result["multi_seed"]["auc_std"]
        multi_f1_mean = multi_result["multi_seed"]["f1_mean"]
        multi_f1_std = multi_result["multi_seed"]["f1_std"]
        seed42_auc = multi_result["multi_seed"]["per_seed_aucs"].get(42, multi_auc_mean)

    # Save eval results
    eval_out = {
        "model": "mlp_position_only",
        "dataset": "tracking_combined",
        "input_dim": INPUT_DIM,
        "velocity_zeroed": VELOCITY_INDICES,
        "n_corners": n_corners,
        "n_matches": n_matches,
        "n_shots": n_shots,
        "auc_mean": multi_auc_mean,
        "auc_std": multi_auc_std,
        "f1_mean": multi_f1_mean,
        "f1_std": multi_f1_std,
        "per_seed_aucs": multi_result["multi_seed"].get("per_seed_aucs", {}),
        "timestamp": str(datetime.now()),
    }
    eval_path = RESULTS_DIR / "tracking_mlp_posonly.json"
    with open(eval_path, "w") as f:
        json.dump(eval_out, f, indent=2, default=str)
    print(f"\nSaved: {eval_path}")

    # Permutation test
    perm_result = None
    if not args.no_permutation and not args.eval_only:
        perm_result = permutation_test_mlp_posonly(
            dataset, n_permutations=args.n_permutations,
            seed=42, device=device, verbose=True,
        )

        perm_result["dataset"] = "tracking_combined"
        perm_result["model"] = "mlp_position_only"
        perm_result["n_corners"] = n_corners
        perm_result["n_matches"] = n_matches
        perm_result["timestamp"] = str(datetime.now())

        perm_path = RESULTS_DIR / "tracking_mlp_posonly_perm.json"
        with open(perm_path, "w") as f:
            json.dump(perm_result, f, indent=2)
        print(f"Saved: {perm_path}")

    # Final report
    print(f"\n{'=' * 60}")
    print("RESULTS: MLP Position-Only on Tracking Data")
    print(f"{'=' * 60}")
    print(f"Corners: {n_corners}")
    print(f"Matches: {n_matches}")
    print(f"Multi-seed AUC ({len(SEEDS)} seeds): {multi_auc_mean:.3f} +/- {multi_auc_std:.3f}")
    print(f"Multi-seed F1  ({len(SEEDS)} seeds): {multi_f1_mean:.3f} +/- {multi_f1_std:.3f}")
    print(f"Seed 42 AUC: {seed42_auc:.3f}")

    if perm_result:
        print(f"\nPermutation test (seed 42):")
        print(f"  Real AUC: {perm_result['real_metric']:.3f}")
        print(f"  Null: {perm_result['null_mean']:.3f} +/- {perm_result['null_std']:.3f}")
        print(f"  p-value: {perm_result['p_value']:.3f}")

    # Comparison table
    perm_p = f"{perm_result['p_value']:.3f}" if perm_result else "---"
    print(f"\n{'=' * 60}")
    print("| MLP condition          | Data        | Velocity? | AUC   | p     |")
    print("|------------------------|-------------|-----------|-------|-------|")
    print(f"| Tracking full (ref)    | 143 SK+DFL  | Yes       | 0.602 | 0.040 |")
    print(f"| Tracking pos-only      | {n_corners} SK+DFL  | No (zeroed) "
          f"| {multi_auc_mean:.3f} | {perm_p} |")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
