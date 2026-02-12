#!/usr/bin/env python3
"""Extract corner kick tracking data from SkillCorner open data.

Usage:
    python -m tracking_extraction.scripts.extract_skillcorner \
        --data-dir /path/to/skillcorner/opendata \
        --output-dir tracking_extraction/output/skillcorner \
        [--match-ids 1886347 1925299]

Prerequisites:
    git clone https://github.com/SkillCorner/opendata.git <data-dir>
    cd <data-dir> && git lfs pull
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking_extraction.skillcorner_adapter import (
    extract_all_matches,
    ALL_MATCH_IDS,
)
from tracking_extraction.core import save_dataset
from tracking_extraction.validate import print_dataset_summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract corner tracking data from SkillCorner open data"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to cloned SkillCorner opendata repo",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="tracking_extraction/output/skillcorner",
        help="Output directory for extracted corners",
    )
    parser.add_argument(
        "--match-ids", type=int, nargs="+", default=None,
        help=f"Match IDs to process (default: all {len(ALL_MATCH_IDS)})",
    )
    parser.add_argument(
        "--pre-seconds", type=float, default=5.0,
        help="Seconds before corner delivery (default: 5.0)",
    )
    parser.add_argument(
        "--post-seconds", type=float, default=5.0,
        help="Seconds after corner delivery (default: 5.0)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        print("Clone the SkillCorner repo first:")
        print(f"  git clone https://github.com/SkillCorner/opendata.git {data_dir}")
        print(f"  cd {data_dir} && git lfs pull")
        sys.exit(1)

    # Extract corners
    corners = extract_all_matches(
        data_dir=data_dir,
        match_ids=args.match_ids,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
    )

    if not corners:
        print("No corners extracted!")
        sys.exit(1)

    # Save dataset
    save_dataset(corners, output_dir)
    print(f"\nSaved {len(corners)} corners to {output_dir}")

    # Print summary
    print_dataset_summary(corners)


if __name__ == "__main__":
    main()
