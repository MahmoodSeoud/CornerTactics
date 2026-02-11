#!/usr/bin/env python
"""Run velocity ablation experiment on DFL corner kick data.

This script:
1. Loads all 7 DFL matches
2. Builds corner kick graph dataset
3. Extracts open-play sequences for pretraining
4. Runs ablation: position-only vs position+velocity
5. Reports results with statistical significance

Usage:
    python scripts/run_ablation.py [--epochs 100] [--pretrain-epochs 50] [--output-dir results/]
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from src.dfl import (
    load_tracking_data,
    load_event_data,
    find_corner_events,
    build_corner_dataset_from_match,
    get_dataset_summary,
    save_corner_dataset,
    CornerKickPredictor,
    extract_open_play_sequences,
    pretrain_spatial_gnn,
    finetune_on_corners,
    zero_out_velocity_features,
    analyze_ablation_results,
    format_ablation_report,
)


def get_all_match_ids(data_dir: Path) -> list:
    """Get all DFL match IDs from the data directory."""
    import glob
    event_files = glob.glob(str(data_dir / "*events*.xml"))
    match_ids = []
    for f in event_files:
        # Extract match ID from filename
        parts = f.split("DFL-MAT-")
        if len(parts) > 1:
            match_id = "DFL-MAT-" + parts[1].replace(".xml", "")
            match_ids.append(match_id)
    return sorted(match_ids)


def build_full_dataset(data_dir: Path, match_ids: list) -> list:
    """Build corner dataset from all matches."""
    full_dataset = []

    for match_id in match_ids:
        print(f"Processing {match_id}...")

        try:
            tracking = load_tracking_data("dfl", data_dir, match_id)
            events = load_event_data("dfl", data_dir, match_id)

            dataset = build_corner_dataset_from_match(
                tracking_dataset=tracking,
                event_dataset=events,
                match_id=match_id,
            )

            print(f"  Found {len(dataset)} corners")
            full_dataset.extend(dataset)

        except Exception as e:
            print(f"  Error processing {match_id}: {e}")
            continue

    return full_dataset


def extract_all_open_play(data_dir: Path, match_ids: list) -> list:
    """Extract open-play sequences from all matches for pretraining."""
    all_sequences = []

    for match_id in match_ids:
        print(f"Extracting open-play from {match_id}...")

        try:
            tracking = load_tracking_data("dfl", data_dir, match_id)
            events = load_event_data("dfl", data_dir, match_id)

            sequences = extract_open_play_sequences(
                tracking, events,
                window_seconds=4.0,
                stride_seconds=8.0,  # Larger stride to reduce dataset size
            )

            print(f"  Extracted {len(sequences)} sequences")
            all_sequences.extend(sequences)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return all_sequences


def run_ablation_experiment(
    corner_dataset: list,
    open_play_sequences: list = None,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
) -> dict:
    """Run the velocity ablation experiment.

    Compares:
    - Condition A: Position-only (vx=0, vy=0)
    - Condition B: Position + Velocity (full features)
    """
    print("\n" + "=" * 60)
    print("VELOCITY ABLATION EXPERIMENT")
    print("=" * 60)

    # Move data to device if using GPU
    def move_to_device(dataset, device):
        """Move all graph tensors to device."""
        for sample in dataset:
            for graph in sample["graphs"]:
                graph.x = graph.x.to(device)
                graph.edge_index = graph.edge_index.to(device)
                if hasattr(graph, "pos") and graph.pos is not None:
                    graph.pos = graph.pos.to(device)
        return dataset

    results = {}

    # Condition A: Position-only
    print("\n--- Condition A: Position-only ---")
    dataset_no_vel = zero_out_velocity_features(corner_dataset)
    dataset_no_vel = move_to_device(dataset_no_vel, device)

    model_no_vel = CornerKickPredictor(node_features=8).to(device)

    if open_play_sequences:
        print("Pretraining spatial GNN...")
        model_no_vel = pretrain_spatial_gnn(
            model_no_vel, open_play_sequences,
            epochs=pretrain_epochs, lr=1e-3
        )

    print("Fine-tuning on corners...")
    results_no_vel = finetune_on_corners(
        model_no_vel, dataset_no_vel,
        epochs=finetune_epochs, lr=lr
    )
    results["position_only"] = results_no_vel

    # Condition B: Position + Velocity
    print("\n--- Condition B: Position + Velocity ---")
    dataset_full = move_to_device(corner_dataset, device)

    model_full = CornerKickPredictor(node_features=8).to(device)

    if open_play_sequences:
        print("Pretraining spatial GNN...")
        model_full = pretrain_spatial_gnn(
            model_full, open_play_sequences,
            epochs=pretrain_epochs, lr=1e-3
        )

    print("Fine-tuning on corners...")
    results_full = finetune_on_corners(
        model_full, dataset_full,
        epochs=finetune_epochs, lr=lr
    )
    results["position_velocity"] = results_full

    return results


def main():
    parser = argparse.ArgumentParser(description="Run velocity ablation experiment")
    parser.add_argument("--data-dir", type=str, default="data/dfl",
                        help="Path to DFL data directory")
    parser.add_argument("--output-dir", type=str, default="results/ablation",
                        help="Output directory for results")
    parser.add_argument("--pretrain-epochs", type=int, default=50,
                        help="Number of pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=100,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pretraining on open-play data")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    start_time = time.time()

    # Get all match IDs
    print("\n--- Loading Data ---")
    match_ids = get_all_match_ids(data_dir)
    print(f"Found {len(match_ids)} matches: {match_ids}")

    # Build corner dataset
    print("\n--- Building Corner Dataset ---")
    corner_dataset = build_full_dataset(data_dir, match_ids)

    summary = get_dataset_summary(corner_dataset)
    print(f"\nDataset Summary:")
    print(f"  Total corners: {summary['total_corners']}")
    print(f"  Shot rate: {summary['shot_rate']:.1%}")
    print(f"  Goal rate: {summary['goal_rate']:.1%}")
    print(f"  Avg frames/corner: {summary['avg_frames_per_corner']:.0f}")

    # Save dataset
    dataset_path = output_dir / "corner_dataset.pkl"
    save_corner_dataset(corner_dataset, dataset_path)
    print(f"Saved dataset to {dataset_path}")

    # Extract open-play sequences for pretraining
    open_play_sequences = None
    if not args.skip_pretrain:
        print("\n--- Extracting Open-Play Sequences ---")
        open_play_sequences = extract_all_open_play(data_dir, match_ids)
        print(f"Total open-play sequences: {len(open_play_sequences)}")

        shot_rate = sum(1 for s in open_play_sequences if s["shot_label"] == 1) / len(open_play_sequences)
        print(f"Shot rate in open-play: {shot_rate:.1%}")

    # Run ablation
    ablation_results = run_ablation_experiment(
        corner_dataset=corner_dataset,
        open_play_sequences=open_play_sequences,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        device=device,
    )

    # Analyze results
    print("\n--- Analyzing Results ---")
    analysis = analyze_ablation_results(ablation_results)

    # Print report
    report = format_ablation_report(analysis)
    print("\n" + report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save analysis as JSON
    analysis_path = output_dir / f"ablation_analysis_{timestamp}.json"
    with open(analysis_path, "w") as f:
        # Convert numpy types for JSON serialization
        json_analysis = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in analysis.items()
            if not isinstance(v, list)  # Skip fold_aucs lists
        }
        json.dump(json_analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    # Save report as text
    report_path = output_dir / f"ablation_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\n\nRuntime: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Saved report to {report_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Runtime: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Results saved to: {output_dir}")

    # Key findings
    print("\nKey Findings:")
    print(f"  Position-only AUC:     {analysis['position_only_mean_auc']:.3f}")
    print(f"  Position+Velocity AUC: {analysis['position_velocity_mean_auc']:.3f}")
    print(f"  Delta:                 {analysis['delta_auc']:.3f}")
    if not np.isnan(analysis.get('p_value', float('nan'))):
        print(f"  p-value:               {analysis['p_value']:.4f}")
        if analysis['p_value'] < 0.05:
            print("  --> SIGNIFICANT: Velocity features improve prediction!")
        elif analysis['p_value'] < 0.10:
            print("  --> Marginally significant improvement")
        else:
            print("  --> Not statistically significant")


if __name__ == "__main__":
    main()
