#!/usr/bin/env python3
"""Convert unified tracking data to USSF-schema PyTorch Geometric graphs.

Usage:
    python -m tracking_extraction.scripts.build_graph_dataset \
        --input-dir tracking_extraction/output/unified \
        --output-path transfer_learning/data/multi_source_corners_dense.pkl \
        [--adjacency dense] [--split]

Produces pickle files compatible with the transfer learning pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking_extraction.core import load_dataset
from tracking_extraction.graph_converter import (
    convert_dataset,
    save_graph_dataset,
    create_splits,
    print_graph_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert unified tracking data to USSF-schema PyG graphs"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to unified JSON dataset directory (must contain manifest.json)",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Output pickle path for full dataset",
    )
    parser.add_argument(
        "--adjacency", type=str, default="dense", choices=["dense", "normal"],
        help="Graph adjacency type (default: dense)",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Also create train/val/test split files alongside output",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Validation set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15,
        help="Test set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits (default: 42)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    if not input_dir.exists() or not (input_dir / "manifest.json").exists():
        print(f"Error: input directory not found or missing manifest.json: {input_dir}")
        sys.exit(1)

    # Load unified dataset
    print(f"Loading corners from {input_dir}...")
    corners = load_dataset(input_dir)
    print(f"Loaded {len(corners)} corners")

    # Convert to graphs
    print(f"Converting to USSF-schema graphs (adjacency={args.adjacency})...")
    dataset = convert_dataset(corners, adjacency=args.adjacency)

    if not dataset:
        print("No graphs produced!")
        sys.exit(1)

    # Save full dataset
    save_graph_dataset(dataset, output_path)
    print(f"\nSaved {len(dataset)} graphs to {output_path}")

    # Print summary
    print_graph_summary(dataset)

    # Create splits if requested
    if args.split:
        print("Creating train/val/test splits...")
        splits = create_splits(
            dataset,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

        stem = output_path.stem
        parent = output_path.parent
        for split_name, split_data in splits.items():
            split_path = parent / f"{stem}_{split_name}.pkl"
            save_graph_dataset(split_data, split_path)
            n_shots = sum(1 for d in split_data if d["labels"]["shot_binary"] == 1)
            print(f"  {split_name}: {len(split_data)} corners "
                  f"({n_shots} shots, {100*n_shots/len(split_data):.0f}%) -> {split_path}")


if __name__ == "__main__":
    main()
