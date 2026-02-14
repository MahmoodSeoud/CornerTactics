#!/usr/bin/env python3
"""Extract corner kick tracking data from SoccerNet GSR pipeline output.

Two-stage workflow:
1. Prepare clip list:
    python -m tracking_extraction.scripts.extract_soccernet_gsr prepare \
        --output-list tracking_extraction/output/gsr_clips.json

2. After running GSR pipeline via SLURM, parse outputs:
    python -m tracking_extraction.scripts.extract_soccernet_gsr parse \
        --gsr-output-dir /path/to/gsr_outputs \
        --clip-list tracking_extraction/output/gsr_clips.json \
        --output-dir tracking_extraction/output/soccernet_gsr
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking_extraction.soccernet_gsr_adapter import (
    prepare_gsr_clip_list,
    process_gsr_outputs,
)
from tracking_extraction.core import save_dataset
from tracking_extraction.validate import print_dataset_summary


def cmd_prepare(args):
    """Prepare clip list for GSR processing."""
    corner_dataset = Path(args.corner_dataset)
    clips_dir = Path(args.clips_dir)
    output_list = Path(args.output_list)

    clips = prepare_gsr_clip_list(
        corner_dataset_json=corner_dataset,
        clips_dir=clips_dir,
        output_list=output_list,
        max_corners=args.max_corners,
    )
    print(f"Prepared {len(clips)} clips -> {output_list}")
    print(f"\nNext: run GSR pipeline on these clips, then use 'parse' command.")


def cmd_parse(args):
    """Parse GSR outputs into unified format."""
    gsr_dir = Path(args.gsr_output_dir)
    clip_list = Path(args.clip_list)
    output_dir = Path(args.output_dir)

    corners = process_gsr_outputs(
        gsr_output_dir=gsr_dir,
        clip_list_path=clip_list,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
    )

    if not corners:
        print("No corners extracted from GSR outputs!")
        sys.exit(1)

    save_dataset(corners, output_dir)
    print(f"\nSaved {len(corners)} corners to {output_dir}")
    print_dataset_summary(corners)


def main():
    parser = argparse.ArgumentParser(
        description="Extract corner tracking from SoccerNet GSR"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prepare command
    prep = subparsers.add_parser("prepare", help="Prepare clip list for GSR")
    prep.add_argument(
        "--corner-dataset", type=str,
        default="FAANTRA/data/corners/corner_dataset.json",
        help="Path to corner_dataset.json",
    )
    prep.add_argument(
        "--clips-dir", type=str,
        default="FAANTRA/data/corners/clips",
        help="Path to extracted video clips",
    )
    prep.add_argument(
        "--output-list", type=str,
        default="tracking_extraction/output/gsr_clips.json",
        help="Output clip list JSON",
    )
    prep.add_argument(
        "--max-corners", type=int, default=None,
        help="Limit number of corners to process",
    )
    prep.add_argument("-v", "--verbose", action="store_true")

    # Parse command
    parse_cmd = subparsers.add_parser("parse", help="Parse GSR outputs")
    parse_cmd.add_argument(
        "--gsr-output-dir", type=str, required=True,
        help="Directory containing GSR JSON outputs",
    )
    parse_cmd.add_argument(
        "--clip-list", type=str, required=True,
        help="Clip list JSON from prepare step",
    )
    parse_cmd.add_argument(
        "--output-dir", type=str,
        default="tracking_extraction/output/soccernet_gsr",
        help="Output directory for extracted corners",
    )
    parse_cmd.add_argument(
        "--pre-seconds", type=float, default=10.0,
    )
    parse_cmd.add_argument(
        "--post-seconds", type=float, default=0.0,
    )
    parse_cmd.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "parse":
        cmd_parse(args)


if __name__ == "__main__":
    main()
