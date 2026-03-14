#!/usr/bin/env python3
"""Permutation test for the USSF position-only ablation condition.

Runs N=100 permutation test on the ussf_position_only ablation (node features
2-5 and edge features 1,4,5 zeroed out) to determine whether position-only
AUC ~0.648 is itself statistically significant or indistinguishable from chance.

Usage:
    cd /home/mseo/CornerTactics
    source FAANTRA/venv/bin/activate
    python scripts/permutation_test_position_only.py
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corner_prediction.config import (
    DATA_DIR,
    N_PERMUTATIONS,
    PRETRAINED_PATH,
    RESULTS_DIR,
    USSF_ABLATION_CONFIGS,
)
from corner_prediction.data.dataset import CornerKickDataset
from corner_prediction.training.ablation import apply_feature_mask
from corner_prediction.training.permutation_test import permutation_test_shot


def main():
    seed = 42
    n_permutations = N_PERMUTATIONS  # 100

    print(f"{'=' * 60}")
    print("Permutation Test: USSF Position-Only Ablation")
    print(f"{'=' * 60}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Seed: {seed}")
    print(f"N permutations: {n_permutations}")
    print(f"Dataset: combined (143 corners, 17 LOMO folds)")

    # Load combined dataset in ussf_aligned mode
    dataset = CornerKickDataset(
        root=str(DATA_DIR),
        records_file="combined_corners.pkl",
        edge_type="dense",
        k=6,
        feature_mode="ussf_aligned",
    )
    print(f"Loaded {len(dataset)} graphs")

    # Apply ussf_position_only feature mask
    config = USSF_ABLATION_CONFIGS["ussf_position_only"]
    print(f"Ablation: ussf_position_only")
    print(f"  Active node features: {config['active_features']}")
    print(f"  Masked node features: [2, 3, 4, 5] (vx_unit, vy_unit, vel_mag, vel_angle)")
    print(f"  Masked edge features: {config['edge_mask_indices']} (speed_diff, vel_sin, vel_cos)")

    masked_dataset = apply_feature_mask(
        dataset,
        active_features=config["active_features"],
        n_features=config["n_features"],
        edge_mask_indices=config.get("edge_mask_indices"),
    )

    # Setup device and lomo_cv kwargs (same as run_permutation in run_all.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pretrained_path = str(PRETRAINED_PATH) if PRETRAINED_PATH.exists() else None
    if pretrained_path is None:
        print("ERROR: USSF pretrained weights not found at", PRETRAINED_PATH)
        sys.exit(1)

    lomo_kwargs = dict(
        backbone_mode="ussf_aligned",
        pretrained_path=pretrained_path,
        freeze=True,
        device=device,
        linear_heads=False,
    )

    # Run permutation test
    result = permutation_test_shot(
        masked_dataset,
        n_permutations=n_permutations,
        seed=seed,
        receiver_mode="oracle",
        verbose=True,
        **lomo_kwargs,
    )

    # Add metadata
    result["ablation"] = "ussf_position_only"
    result["dataset"] = "combined"
    result["timestamp"] = str(datetime.now())

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "combined_perm_shot_ussf_position_only.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    pkl_path = RESULTS_DIR / "combined_perm_shot_ussf_position_only.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved pickle: {pkl_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Real AUC (position-only, oracle): {result['real_metric']:.4f}")
    print(f"Null distribution: mean={result['null_mean']:.4f}, std={result['null_std']:.4f}")
    print(f"p-value: {result['p_value']:.4f} {'***' if result['p_value'] < 0.01 else '**' if result['p_value'] < 0.05 else '(not significant)'}")
    print(f"Significant (p < 0.05): {result['significant']}")


if __name__ == "__main__":
    main()
