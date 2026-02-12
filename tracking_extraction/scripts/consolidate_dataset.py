#!/usr/bin/env python3
"""Consolidate tracking data from all sources into unified dataset.

Usage:
    python -m tracking_extraction.scripts.consolidate_dataset \
        --skillcorner-dir tracking_extraction/output/skillcorner \
        --dfl-dir tracking_extraction/output/dfl \
        --gsr-dir tracking_extraction/output/soccernet_gsr \
        --output-dir tracking_extraction/output/unified \
        --min-players 15 \
        --min-frames 50
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking_extraction.core import load_dataset, save_dataset
from tracking_extraction.validate import (
    compute_quality_metrics,
    validate_corner,
    print_dataset_summary,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge all tracking sources into unified training dataset"
    )
    parser.add_argument(
        "--skillcorner-dir", type=str, default=None,
        help="SkillCorner output directory",
    )
    parser.add_argument(
        "--dfl-dir", type=str, default=None,
        help="DFL output directory",
    )
    parser.add_argument(
        "--gsr-dir", type=str, default=None,
        help="SoccerNet GSR output directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="tracking_extraction/output/unified",
        help="Output directory for unified dataset",
    )
    parser.add_argument(
        "--min-players", type=float, default=15.0,
        help="Minimum mean players per frame to include (default: 15)",
    )
    parser.add_argument(
        "--min-frames", type=int, default=50,
        help="Minimum frames per corner to include (default: 50)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load from each source
    all_corners = []
    source_counts = {}

    for name, dir_path in [
        ("skillcorner", args.skillcorner_dir),
        ("dfl", args.dfl_dir),
        ("soccernet_gsr", args.gsr_dir),
    ]:
        if dir_path is None:
            continue
        p = Path(dir_path)
        if not p.exists() or not (p / "manifest.json").exists():
            logger.warning("Skipping %s: directory or manifest not found at %s", name, p)
            continue

        corners = load_dataset(p)
        source_counts[name] = len(corners)
        all_corners.extend(corners)
        logger.info("Loaded %d corners from %s", len(corners), name)

    if not all_corners:
        print("No corners loaded from any source!")
        sys.exit(1)

    print(f"\nLoaded {len(all_corners)} corners total")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    # Apply quality filters
    filtered = []
    rejected_reasons = {"low_players": 0, "few_frames": 0, "warnings": 0}

    for corner in all_corners:
        metrics = compute_quality_metrics(corner)

        if metrics["n_frames"] < args.min_frames:
            rejected_reasons["few_frames"] += 1
            continue

        if metrics["mean_players_per_frame"] < args.min_players:
            rejected_reasons["low_players"] += 1
            continue

        # Check for critical warnings (out-of-bounds, extreme velocities)
        warnings = validate_corner(corner)
        critical = [w for w in warnings if "out of bounds" in w or "too fast" in w]
        if len(critical) > 5:
            rejected_reasons["warnings"] += 1
            continue

        filtered.append(corner)

    print(f"\nAfter quality filtering: {len(filtered)} / {len(all_corners)} corners")
    for reason, count in rejected_reasons.items():
        if count > 0:
            print(f"  Rejected ({reason}): {count}")

    # Check for duplicate corner IDs
    ids = [c.corner_id for c in filtered]
    if len(ids) != len(set(ids)):
        dupes = [cid for cid in ids if ids.count(cid) > 1]
        logger.warning("Duplicate corner IDs found: %s", set(dupes))

    # Save unified dataset
    output_dir = Path(args.output_dir)
    save_dataset(filtered, output_dir)
    print(f"\nSaved {len(filtered)} corners to {output_dir}")

    # Print summary
    print_dataset_summary(filtered)


if __name__ == "__main__":
    main()
