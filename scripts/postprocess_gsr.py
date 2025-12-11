#!/usr/bin/env python3
"""Post-process GSR JSON outputs to extract player positions and velocities.

Phase 5: Post-Processing
- Parse GSR JSON outputs
- Extract player positions (bbox_pitch coordinates)
- Compute velocities from frame-to-frame positions
- Create corner snapshots at t=0 (frame 50)
- Merge with corner metadata
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def parse_gsr_json(json_path: Path) -> pd.DataFrame:
    """Parse a single GSR JSON file into a DataFrame.

    Returns DataFrame with columns:
        frame, track_id, x, y, role, team, jersey
    """
    with open(json_path) as f:
        data = json.load(f)

    records = []
    for pred in data.get('predictions', []):
        # image_id is the frame number
        frame = pred.get('image_id', 0)

        # bbox_pitch contains pitch coordinates
        bbox_pitch = pred.get('bbox_pitch', {})
        x = bbox_pitch.get('x_bottom_middle')
        y = bbox_pitch.get('y_bottom_middle')

        if x is None or y is None:
            continue

        records.append({
            'frame': frame,
            'track_id': pred.get('track_id', -1),
            'x': x,
            'y': y,
            'role': pred.get('role', 'unknown'),
            'team': pred.get('team', 'unknown'),
            'jersey': pred.get('jersey_number', -1)
        })

    return pd.DataFrame(records)


def compute_velocities(df: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    """Compute velocities from position data.

    Args:
        df: DataFrame with columns [frame, track_id, x, y, ...]
        fps: Frames per second

    Returns:
        DataFrame with additional columns [vx, vy, speed]
    """
    dt = 1.0 / fps

    # Sort by track_id and frame
    df = df.sort_values(['track_id', 'frame']).copy()

    # Compute velocities within each track
    df['vx'] = df.groupby('track_id')['x'].diff() / dt
    df['vy'] = df.groupby('track_id')['y'].diff() / dt
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)

    # Fill NaN for first frame of each track
    df[['vx', 'vy', 'speed']] = df[['vx', 'vy', 'speed']].fillna(0)

    return df


def extract_snapshot(df: pd.DataFrame, target_frame: int = 50) -> pd.DataFrame:
    """Extract player positions at a specific frame (t=0).

    Args:
        df: DataFrame with tracking data
        target_frame: Frame number to extract (default 50 = 2 seconds into 10-sec clip at 25fps)

    Returns:
        DataFrame with snapshot at target frame
    """
    # Find closest frame if exact frame not available
    available_frames = df['frame'].unique()
    if target_frame in available_frames:
        return df[df['frame'] == target_frame].copy()
    else:
        # Find closest frame
        closest = available_frames[np.abs(available_frames - target_frame).argmin()]
        return df[df['frame'] == closest].copy()


def process_all_corners(json_dir: Path, output_dir: Path, metadata_path: Path):
    """Process all GSR JSON files and create final outputs."""

    # Load corner metadata
    with open(metadata_path) as f:
        metadata = {c['corner_id']: c for c in json.load(f)}

    # Find all JSON files
    json_files = sorted(json_dir.glob('CORNER-*.json'))
    print(f"Found {len(json_files)} GSR JSON files")

    all_snapshots = []

    for json_file in tqdm(json_files, desc="Processing GSR outputs"):
        # Extract corner ID from filename
        corner_id = int(json_file.stem.replace('CORNER-', ''))

        try:
            # Parse JSON
            df = parse_gsr_json(json_file)

            if df.empty:
                print(f"Warning: No predictions in {json_file}")
                continue

            # Compute velocities
            df = compute_velocities(df)

            # Extract snapshot at t=0
            snapshot = extract_snapshot(df)
            snapshot['corner_id'] = corner_id

            all_snapshots.append(snapshot)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    if not all_snapshots:
        print("No snapshots extracted!")
        return

    # Combine all snapshots
    snapshots_df = pd.concat(all_snapshots, ignore_index=True)
    print(f"Total snapshots: {len(snapshots_df)}")
    print(f"Unique corners: {snapshots_df['corner_id'].nunique()}")

    # Add metadata
    meta_cols = ['game_path', 'half', 'timestamp_seconds', 'corner_team',
                 'competition', 'season', 'home_team', 'away_team']
    for col in meta_cols:
        snapshots_df[col] = snapshots_df['corner_id'].map(
            lambda cid: metadata.get(cid, {}).get(col, None)
        )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parquet format (efficient)
    parquet_path = output_dir / 'corner_snapshots.parquet'
    snapshots_df.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path}")

    # CSV for inspection
    csv_path = output_dir / 'corner_snapshots.csv'
    snapshots_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Corners processed: {snapshots_df['corner_id'].nunique()}")
    print(f"  Total player detections: {len(snapshots_df)}")
    print(f"  Avg players per corner: {len(snapshots_df) / snapshots_df['corner_id'].nunique():.1f}")

    # Check for corners with few detections
    players_per_corner = snapshots_df.groupby('corner_id').size()
    low_detection = (players_per_corner < 5).sum()
    print(f"  Corners with < 5 players: {low_detection}")


def main():
    parser = argparse.ArgumentParser(description='Post-process GSR JSON outputs')
    parser.add_argument('--json-dir', type=str,
                        default='/home/mseo/CornerTactics/outputs/json',
                        help='Directory containing GSR JSON outputs')
    parser.add_argument('--output-dir', type=str,
                        default='/home/mseo/CornerTactics/outputs/processed',
                        help='Output directory for processed data')
    parser.add_argument('--metadata', type=str,
                        default='/home/mseo/CornerTactics/data/processed/corner_metadata.json',
                        help='Path to corner metadata JSON')
    args = parser.parse_args()

    process_all_corners(
        json_dir=Path(args.json_dir),
        output_dir=Path(args.output_dir),
        metadata_path=Path(args.metadata)
    )


if __name__ == '__main__':
    main()
