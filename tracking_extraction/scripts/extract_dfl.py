#!/usr/bin/env python3
"""Extract DFL corner kick tracking data to unified format.

Usage:
    python -m tracking_extraction.scripts.extract_dfl \
        --data-dir data/dfl \
        --output-dir tracking_extraction/output/dfl

Processes all 7 DFL matches and converts to unified CornerTrackingData JSON.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking_extraction.dfl_adapter import convert_dfl_from_paths
from tracking_extraction.core import save_dataset
from tracking_extraction.validate import print_dataset_summary

logger = logging.getLogger(__name__)


def find_match_ids(data_dir: Path):
    """Extract unique match IDs from DFL XML filenames."""
    pattern = re.compile(r"DFL-MAT-\w+")
    match_ids = set()
    for f in data_dir.glob("*.xml"):
        m = pattern.search(f.name)
        if m:
            match_ids.add(m.group())
    return sorted(match_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DFL corners to unified tracking format"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/dfl",
        help="Directory containing DFL XML files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="tracking_extraction/output/dfl",
        help="Output directory for unified JSON",
    )
    parser.add_argument(
        "--pre-seconds", type=float, default=5.0,
        help="Seconds before corner delivery (default: 5.0)",
    )
    parser.add_argument(
        "--post-seconds", type=float, default=5.0,
        help="Seconds after corner delivery (default: 5.0)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: DFL data directory not found: {data_dir}")
        sys.exit(1)

    match_ids = find_match_ids(data_dir)
    print(f"Found {len(match_ids)} DFL matches: {match_ids}")

    all_corners = []
    for mid in match_ids:
        try:
            corners = convert_dfl_from_paths(
                data_dir=data_dir,
                match_id=mid,
                pre_seconds=args.pre_seconds,
                post_seconds=args.post_seconds,
            )
            all_corners.extend(corners)
            print(f"  {mid}: {len(corners)} corners")
        except Exception:
            logger.exception("Failed to process match %s", mid)

    if not all_corners:
        print("No corners extracted!")
        sys.exit(1)

    # Save dataset
    save_dataset(all_corners, output_dir)
    print(f"\nSaved {len(all_corners)} corners to {output_dir}")

    # Print summary
    print_dataset_summary(all_corners)

    # Cross-validate against expected count
    print(f"\nCross-validation:")
    print(f"  Expected: 57 corners from 7 matches")
    print(f"  Got: {len(all_corners)} corners from {len(match_ids)} matches")
    shots = sum(1 for c in all_corners if c.outcome == "shot")
    print(f"  Shots: {shots}, No-shot: {len(all_corners) - shots}")


if __name__ == "__main__":
    main()
