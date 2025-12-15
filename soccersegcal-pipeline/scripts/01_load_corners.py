#!/usr/bin/env python3
"""
Step 1: Load corner kick data and map to existing video clips.

This script reads the existing corner metadata and creates an index
mapping corner IDs to video clips for the pipeline.
"""

import pandas as pd
import json
from pathlib import Path
import argparse


def load_corners(
    corners_csv: str = "/home/mseo/CornerTactics/data/processed/soccernet_corners.csv",
    clips_dir: str = "/home/mseo/CornerTactics/data/corner_clips"
) -> pd.DataFrame:
    """
    Load corner annotations and match with available video clips.

    Args:
        corners_csv: Path to soccernet_corners.csv
        clips_dir: Directory containing extracted corner clips

    Returns:
        DataFrame with corner info and clip paths
    """

    # Load corner annotations
    corners_df = pd.read_csv(corners_csv)
    print(f"Loaded {len(corners_df)} corner annotations")

    # Find available clips
    clips_path = Path(clips_dir)
    available_clips = {
        int(p.stem.split('_')[1]): str(p)
        for p in clips_path.glob("corner_*.mp4")
    }
    print(f"Found {len(available_clips)} video clips")

    # Map clips to corners
    corners_df['clip_path'] = corners_df['corner_id'].map(available_clips)
    corners_df['has_clip'] = corners_df['clip_path'].notna()

    # Filter to only visible corners with clips
    visible_with_clips = corners_df[
        (corners_df['visibility'] == 'visible') &
        (corners_df['has_clip'])
    ].copy()

    print(f"Visible corners with clips: {len(visible_with_clips)}")

    return visible_with_clips


def save_corner_index(
    corners_df: pd.DataFrame,
    output_file: str
):
    """Save corner index for pipeline processing."""

    # Create index with essential columns
    index_df = corners_df[[
        'corner_id', 'game_path', 'half', 'timestamp_seconds',
        'position_ms', 'team', 'visibility', 'clip_path'
    ]].copy()

    index_df.to_csv(output_file, index=False)
    print(f"Saved corner index to {output_file}")

    # Also save as JSON for easier loading
    json_output = output_file.replace('.csv', '.json')
    index_df.to_json(json_output, orient='records', indent=2)
    print(f"Saved corner index to {json_output}")

    return index_df


def main():
    parser = argparse.ArgumentParser(description='Load corner kick data')
    parser.add_argument('--corners-csv',
                        default='/home/mseo/CornerTactics/data/processed/soccernet_corners.csv',
                        help='Path to corners CSV')
    parser.add_argument('--clips-dir',
                        default='/home/mseo/CornerTactics/data/corner_clips',
                        help='Directory with corner video clips')
    parser.add_argument('--output',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_index.csv',
                        help='Output index file')
    args = parser.parse_args()

    # Load corners
    corners_df = load_corners(args.corners_csv, args.clips_dir)

    # Save index
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_corner_index(corners_df, args.output)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total corners: {len(corners_df)}")
    print(f"By team: {corners_df['team'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
