#!/usr/bin/env python3
"""Extract 10-second video clips for each corner kick.

For each corner: extract clip from 2 sec before to 8 sec after the kick.
Input: {game_path}/{half}_720p.mkv
Output: corner_clips/corner_{id:04d}.mp4
"""

import json
import subprocess
import csv
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import argparse


# Find ffmpeg - check common locations
FFMPEG_PATH = shutil.which('ffmpeg') or os.environ.get('FFMPEG_PATH', '/home/mseo/.conda/envs/robo/bin/ffmpeg')


def ms_to_seconds(ms: int) -> float:
    """Convert milliseconds to seconds."""
    return ms / 1000.0


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_ms: int,
    duration_sec: float = 10.0,
    offset_before_sec: float = 2.0
) -> bool:
    """Extract a clip from video using ffmpeg.

    Args:
        video_path: Path to source video
        output_path: Path for output clip
        start_ms: Corner event time in milliseconds
        duration_sec: Total clip duration
        offset_before_sec: How many seconds before the event to start

    Returns:
        True if successful, False otherwise
    """
    # Calculate start time (2 seconds before corner)
    start_sec = max(0, ms_to_seconds(start_ms) - offset_before_sec)

    cmd = [
        FFMPEG_PATH,
        '-y',  # Overwrite output
        '-ss', str(start_sec),  # Seek to start (fast seek before input)
        '-i', str(video_path),
        '-t', str(duration_sec),  # Duration
        '-c:v', 'libopenh264',  # Video codec (using libopenh264 from conda ffmpeg)
        '-an',  # No audio
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"\nFFmpeg error for {output_path}:")
            print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout extracting {output_path}")
        return False
    except Exception as e:
        print(f"Error extracting {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract corner kick video clips')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--visible-only', action='store_true', help='Only extract visible corners')
    args = parser.parse_args()

    # Load corner metadata
    metadata_path = Path('/home/mseo/CornerTactics/data/processed/corner_metadata.json')
    with open(metadata_path) as f:
        corners = json.load(f)

    print(f"Loaded {len(corners)} corners")

    # Filter by visibility if requested
    if args.visible_only:
        corners = [c for c in corners if c['visibility'] == 'visible']
        print(f"Filtered to {len(corners)} visible corners")

    # Slice by index range
    corners = corners[args.start:args.end]
    print(f"Processing corners {args.start} to {args.start + len(corners) - 1}")

    # Setup paths
    soccernet_dir = Path('/home/mseo/CornerTactics/data/misc/soccernet/videos')
    output_dir = Path('/home/mseo/CornerTactics/data/corner_clips')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    success = 0
    failed = 0
    skipped = 0
    missing_video = []

    for corner in tqdm(corners, desc="Extracting clips"):
        corner_id = corner['corner_id']
        output_path = output_dir / f"corner_{corner_id:04d}.mp4"

        # Skip if already exists
        if output_path.exists():
            skipped += 1
            continue

        # Find source video
        half = corner['half']
        video_filename = f"{half}_720p.mkv"
        video_path = soccernet_dir / corner['game_path'] / video_filename

        if not video_path.exists():
            missing_video.append(str(video_path))
            failed += 1
            continue

        if args.dry_run:
            print(f"Would extract: {video_path} -> {output_path}")
            print(f"  Time: {corner['position_ms']}ms ({corner['timestamp_seconds']}s)")
            continue

        # Extract clip
        if extract_clip(video_path, output_path, corner['position_ms']):
            success += 1
        else:
            failed += 1

    # Print summary
    print(f"\nExtraction complete:")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (exists): {skipped}")

    if missing_video:
        print(f"\nMissing videos ({len(set(missing_video))} unique):")
        for v in sorted(set(missing_video))[:10]:
            print(f"  {v}")
        if len(set(missing_video)) > 10:
            print(f"  ... and {len(set(missing_video)) - 10} more")

    # Save extraction status
    status_path = output_dir / 'extraction_status.json'
    with open(status_path, 'w') as f:
        json.dump({
            'total_corners': len(corners),
            'successful': success,
            'failed': failed,
            'skipped': skipped,
            'missing_videos': list(set(missing_video))
        }, f, indent=2)


if __name__ == '__main__':
    main()
