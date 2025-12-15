#!/usr/bin/env python3
"""
Step 2: Extract frames from corner video clips at specific time offsets.

Extracts frames at:
- 0ms: Corner kick moment
- 2000ms: Ball in flight / delivery
- 5000ms: Outcome moment

These frames will be used for camera calibration and player detection.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


def extract_frames_from_clip(
    clip_path: str,
    corner_id: int,
    output_dir: str,
    offsets_ms: list = [0, 2000, 5000]
) -> list:
    """
    Extract frames at specified offsets from a corner video clip.

    Args:
        clip_path: Path to corner video clip
        corner_id: Corner ID for naming
        output_dir: Directory to save frames
        offsets_ms: Time offsets in milliseconds

    Returns:
        List of dicts with frame info
    """

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = (total_frames / fps) * 1000 if fps > 0 else 0

    frame_records = []

    for offset in offsets_ms:
        # Skip if offset exceeds video duration
        if offset >= duration_ms:
            continue

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_MSEC, offset)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Save frame
        frame_filename = f"corner_{corner_id:04d}_offset_{offset:05d}.jpg"
        frame_path = Path(output_dir) / frame_filename
        cv2.imwrite(str(frame_path), frame)

        frame_records.append({
            'corner_id': corner_id,
            'offset_ms': offset,
            'frame_path': str(frame_path),
            'width': frame.shape[1],
            'height': frame.shape[0],
            'fps': fps
        })

    cap.release()
    return frame_records


def extract_all_frames(
    corners_df: pd.DataFrame,
    output_dir: str,
    offsets_ms: list = [0, 2000, 5000],
    limit: int = None
) -> pd.DataFrame:
    """
    Extract frames from all corner clips.

    Args:
        corners_df: DataFrame with corner info and clip paths
        output_dir: Directory to save frames
        offsets_ms: Time offsets to extract
        limit: Optional limit on number of corners to process

    Returns:
        DataFrame with frame info
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if limit:
        corners_df = corners_df.head(limit)

    all_frames = []

    for _, row in tqdm(corners_df.iterrows(), total=len(corners_df), desc="Extracting frames"):
        corner_id = row['corner_id']
        clip_path = row['clip_path']

        if pd.isna(clip_path) or not Path(clip_path).exists():
            continue

        frames = extract_frames_from_clip(
            clip_path,
            corner_id,
            output_dir,
            offsets_ms
        )

        # Add corner metadata to each frame record
        for frame in frames:
            frame['game_path'] = row['game_path']
            frame['team'] = row['team']

        all_frames.extend(frames)

    return pd.DataFrame(all_frames)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from corner clips')
    parser.add_argument('--corners-index',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_index.csv',
                        help='Path to corner index CSV')
    parser.add_argument('--output-dir',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/frames',
                        help='Directory to save extracted frames')
    parser.add_argument('--offsets',
                        type=int,
                        nargs='+',
                        default=[0, 2000, 5000],
                        help='Time offsets in ms')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit number of corners to process')
    parser.add_argument('--output-index',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/frames_index.csv',
                        help='Output frame index file')
    args = parser.parse_args()

    # Load corner index
    corners_df = pd.read_csv(args.corners_index)
    print(f"Loaded {len(corners_df)} corners")

    # Extract frames
    frames_df = extract_all_frames(
        corners_df,
        args.output_dir,
        args.offsets,
        args.limit
    )

    # Save frame index
    frames_df.to_csv(args.output_index, index=False)
    print(f"\nExtracted {len(frames_df)} frames")
    print(f"Saved index to {args.output_index}")

    # Also save as JSON
    json_output = args.output_index.replace('.csv', '.json')
    frames_df.to_json(json_output, orient='records', indent=2)

    # Summary
    print("\n=== Summary ===")
    print(f"Total frames: {len(frames_df)}")
    print(f"Unique corners: {frames_df['corner_id'].nunique()}")
    print(f"Offsets extracted: {frames_df['offset_ms'].unique().tolist()}")


if __name__ == "__main__":
    main()
