#!/usr/bin/env python3
"""Prepare batched symlink directories of corner clips for TrackLab GSR processing.

Creates batch directories of symlinks pointing to the original corner clip .mp4 files.
Each batch can be processed by a single SLURM array task.

Usage:
    # Pilot: 50 clips in 5 batches of 10
    python -m tracking_extraction.scripts.prepare_gsr_batches --pilot

    # Full: all visible corners in batches of 50
    python -m tracking_extraction.scripts.prepare_gsr_batches --batch-size 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_batches(
    corner_dataset_json: Path,
    clips_dir: Path,
    output_dir: Path,
    batch_size: int = 50,
    max_corners: int | None = None,
) -> dict:
    """Create batch directories with symlinks to corner clip videos.

    Args:
        corner_dataset_json: Path to FAANTRA/data/corners/corner_dataset.json
        clips_dir: Path to FAANTRA/data/corners/clips/
        output_dir: Where to create batch directories
        batch_size: Number of clips per batch
        max_corners: Limit total corners (None = all visible)

    Returns:
        Manifest dict with corner-to-batch mapping
    """
    with open(corner_dataset_json) as f:
        dataset = json.load(f)

    # Collect visible corners with existing clips
    corners = []
    for i, corner in enumerate(dataset["corners"]):
        if corner.get("visibility") != "visible":
            continue

        corner_id = f"corner_{i:04d}"
        clip_path = clips_dir / corner_id / "720p.mp4"

        if not clip_path.exists():
            continue

        corners.append({
            "corner_id": corner_id,
            "corner_idx": i,
            "clip_path": str(clip_path.resolve()),
            "match_dir": corner["match_dir"],
            "outcome": corner["outcome"],
        })

        if max_corners and len(corners) >= max_corners:
            break

    logger.info("Found %d visible corners with clips", len(corners))

    # Create batch directories
    output_dir = Path(output_dir)
    if output_dir.exists():
        # Clean existing batches
        for batch_dir in output_dir.glob("batch_*"):
            for link in batch_dir.iterdir():
                link.unlink()
            batch_dir.rmdir()

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "total_corners": len(corners),
        "batch_size": batch_size,
        "num_batches": 0,
        "corners": {},  # corner_id -> {batch_idx, clip_path, ...}
    }

    batch_idx = 0
    for i, corner in enumerate(corners):
        if i % batch_size == 0 and i > 0:
            batch_idx += 1

        batch_dir = output_dir / f"batch_{batch_idx:04d}"
        batch_dir.mkdir(exist_ok=True)

        # Symlink: batch_dir/corner_XXXX.mp4 -> original clip
        link_name = batch_dir / f"{corner['corner_id']}.mp4"
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        os.symlink(corner["clip_path"], link_name)

        manifest["corners"][corner["corner_id"]] = {
            "batch_idx": batch_idx,
            "corner_idx": corner["corner_idx"],
            "clip_path": corner["clip_path"],
            "match_dir": corner["match_dir"],
            "outcome": corner["outcome"],
        }

    manifest["num_batches"] = batch_idx + 1

    # Save manifest
    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Created %d batches (%d clips/batch) in %s",
        manifest["num_batches"],
        batch_size,
        output_dir,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare corner clip batches for GSR processing"
    )
    parser.add_argument(
        "--corner-dataset",
        type=str,
        default="FAANTRA/data/corners/corner_dataset.json",
        help="Path to corner_dataset.json",
    )
    parser.add_argument(
        "--clips-dir",
        type=str,
        default="FAANTRA/data/corners/clips",
        help="Path to extracted video clips",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sn-gamestate/data/corner_batches",
        help="Output directory for batch symlinks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Clips per batch (default: 50)",
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=None,
        help="Limit number of corners",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: 50 clips, 10 per batch",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.pilot:
        args.batch_size = 10
        args.max_corners = 50

    manifest = prepare_batches(
        corner_dataset_json=Path(args.corner_dataset),
        clips_dir=Path(args.clips_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_corners=args.max_corners,
    )

    print(f"Total corners: {manifest['total_corners']}")
    print(f"Batches: {manifest['num_batches']} (size {manifest['batch_size']})")
    print(f"Manifest: {Path(args.output_dir) / 'batch_manifest.json'}")


if __name__ == "__main__":
    main()
