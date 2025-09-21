#!/usr/bin/env python3
"""
Command-line interface for corner frame extraction.
"""

import argparse
import sys
import logging
from pathlib import Path

from .corner_frame_pipeline import CornerFramePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract single frames at corner kick moments from SoccerNet videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames from all games in data directory
  python src/cli.py --data-dir ./data

  # Extract frames with custom output CSV
  python src/cli.py --data-dir ./data --output ./corner_frames.csv
        """
    )

    parser.add_argument('--data-dir', required=True,
                       help='Root data directory containing SoccerNet data')
    parser.add_argument('--output',
                       help='Output CSV file path (default: data_dir/insights/corner_frames_metadata.csv)')
    parser.add_argument('--split-by-visibility', action='store_true',
                       help='Split frames into visible/ and not_shown/ subdirectories')

    args = parser.parse_args()

    try:
        # Create and run pipeline
        pipeline = CornerFramePipeline(args.data_dir, args.output, split_by_visibility=args.split_by_visibility)
        result_csv = pipeline.extract_all_corners()

        print(f"Corner frame extraction complete!")
        print(f"Results saved to: {result_csv}")

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()