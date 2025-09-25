#!/usr/bin/env python3
"""
Extract corner kick video clips from SoccerNet videos.

Carmack-style: Simple functions that do one thing well.
"""

import argparse
import csv
import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional

from src.data_loader import SoccerNetDataLoader

# Constants - no magic strings
CORNER_CLIPS_DIR = "datasets/soccernet/corner_clips"
VIDEO_QUALITY = "720p"
FFMPEG_TIMEOUT = 60  # Longer timeout for video clips
CLIP_DURATION = 20   # 20 second clips
CLIP_START_OFFSET = -5  # Start 5 seconds before corner kick

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_game_time(time_string: str) -> Optional[int]:
    """Parse SoccerNet time format to seconds. Returns None on error."""
    try:
        # Handle "1 - 05:30" (v2) or "05:30" (v3) formats
        time_part = time_string.split(' - ')[-1]
        minutes, seconds = map(int, time_part.split(':'))
        return max(0, minutes * 60 + seconds)  # Ensure non-negative
    except (ValueError, IndexError):
        logger.error(f"Cannot parse time: {time_string}")
        return None


def find_video_file(data_dir: str, game_path: str, half_number: int) -> Optional[Path]:
    """Find the video file for this game and half."""
    video_path = Path(data_dir) / game_path / f"{half_number}_{VIDEO_QUALITY}.mkv"

    if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        return None

    return video_path


def create_clip_filename(game_path: str, half: int, seconds: int, team_name: str) -> str:
    """Create a unique filename for the extracted video clip."""
    # Get meaningful game identifier
    path_parts = game_path.split('/')
    game_id = '_'.join(path_parts[-3:] if len(path_parts) >= 3 else path_parts)

    # Sanitize for filesystem
    clean_game_id = ''.join(c for c in game_id if c.isalnum() or c in '_-')[:40]
    clean_team = ''.join(c for c in team_name if c.isalnum() or c in '_')

    minutes, secs = divmod(seconds, 60)
    return f"{clean_game_id}_{half}H_{minutes:02d}m{secs:02d}s_{clean_team}.mp4"


def get_output_directory(data_dir: str, visibility: str, split_by_visibility: bool) -> Path:
    """Determine where to save the video clip based on visibility."""
    base_dir = Path(data_dir) / CORNER_CLIPS_DIR

    if not split_by_visibility:
        return base_dir

    if visibility == 'visible':
        return base_dir / "visible"
    elif visibility == 'not shown':
        return base_dir / "not_shown"
    else:
        return base_dir / "visible"  # Default for unknown


def extract_clip_with_ffmpeg(video_path: Path, start_seconds: int, duration: int, output_path: Path) -> bool:
    """Run ffmpeg to extract a video clip. Returns True on success."""
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-ss', str(start_seconds),  # Start time
        '-i', str(video_path),      # Input video
        '-t', str(duration),        # Duration
        '-c:v', 'libx264',          # H.264 codec
        '-c:a', 'aac',              # AAC audio codec
        '-crf', '23',               # Good quality
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr.decode().strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return False


def extract_corner_clip(data_dir: str, game_path: str, game_time: str,
                       team: str, half: int, visibility: str,
                       split_by_visibility: bool = False) -> Optional[str]:
    """Extract one corner video clip. Returns output path or None."""

    # Parse the corner kick timestamp
    corner_timestamp = parse_game_time(game_time)
    if corner_timestamp is None:
        return None

    # Calculate clip start time (5 seconds before corner kick)
    clip_start_time = max(0, corner_timestamp + CLIP_START_OFFSET)

    # Find the video file
    video_path = find_video_file(data_dir, game_path, half)
    if video_path is None:
        return None

    # Setup output path
    output_dir = get_output_directory(data_dir, visibility, split_by_visibility)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = create_clip_filename(game_path, half, corner_timestamp, team)
    output_path = output_dir / filename

    # Extract the video clip
    success = extract_clip_with_ffmpeg(video_path, clip_start_time, CLIP_DURATION, output_path)

    if success:
        logger.info(f"Extracted clip: {filename}")
        return str(output_path)

    return None


def setup_output_csv(data_dir: str, output_csv: str = None) -> Path:
    """Setup and return the output CSV path."""
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return csv_path

    output_dir = Path(data_dir) / "insights"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "corner_clips_metadata.csv"


def process_corner(csv_writer, data_dir: str, game_path: str, corner: dict,
                  split_by_visibility: bool) -> bool:
    """Process one corner and write to CSV. Returns True if extraction succeeded."""

    clip_path = extract_corner_clip(
        data_dir, game_path,
        corner['gameTime'],
        corner.get('team', 'unknown'),
        corner['half'],
        corner.get('visibility', 'unknown'),
        split_by_visibility
    )

    # Write to CSV regardless of extraction success
    csv_writer.writerow({
        'game_path': game_path,
        'game_time': corner['gameTime'],
        'half': str(corner['half']),
        'team': corner.get('team', 'unknown'),
        'visibility': corner.get('visibility', 'unknown'),
        'clip_path': clip_path or ''
    })

    return clip_path is not None


def process_game(csv_writer, data_loader, data_dir: str, game_path: str,
                split_by_visibility: bool) -> tuple[int, int]:
    """Process all corners in one game. Returns (total_corners, successful_extractions)."""

    corners = data_loader.get_corner_events(game_path)
    if not corners:
        return 0, 0

    logger.info(f"Processing {len(corners)} corners in {game_path}")

    successful_count = 0
    for corner in corners:
        if process_corner(csv_writer, data_dir, game_path, corner, split_by_visibility):
            successful_count += 1

    return len(corners), successful_count


def extract_all_corners(data_dir: str, output_csv: str = None, split_by_visibility: bool = False) -> str:
    """Extract 20-second video clips for all corners and generate metadata CSV."""

    # Setup
    data_loader = SoccerNetDataLoader(data_dir)
    csv_path = setup_output_csv(data_dir, output_csv)
    games = data_loader.list_games()

    if not games:
        logger.warning("No games found")
        return str(csv_path)

    logger.info(f"Processing {len(games)} games")

    # Process all games
    total_corners = 0
    total_successful = 0

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['game_path', 'game_time', 'half', 'team', 'visibility', 'clip_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, game_path in enumerate(games, 1):
            logger.info(f"Game {i}/{len(games)}: {game_path}")

            try:
                corner_count, success_count = process_game(
                    writer, data_loader, data_dir, game_path, split_by_visibility
                )
                total_corners += corner_count
                total_successful += success_count

            except Exception as e:
                logger.error(f"Failed processing {game_path}: {e}")

    # Summary
    success_rate = (total_successful / total_corners * 100) if total_corners > 0 else 0
    logger.info(f"Extracted {total_successful}/{total_corners} clips ({success_rate:.1f}%)")
    logger.info(f"Results: {csv_path}")

    return str(csv_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract 20-second video clips around corner kick moments from SoccerNet videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract clips from all games in data directory
  python scripts/extract_corners.py --data-dir ./data

  # Extract clips with custom output CSV
  python scripts/extract_corners.py --data-dir ./data --output ./corner_clips.csv

  # Split clips by visibility (visible/not_shown directories)
  python scripts/extract_corners.py --data-dir ./data --split-by-visibility

Clip details:
  - Duration: 20 seconds per clip
  - Start time: 5 seconds before corner kick moment
  - Output format: MP4 (H.264/AAC)
        """
    )

    parser.add_argument('--data-dir', required=True,
                       help='Root data directory containing SoccerNet data')
    parser.add_argument('--output',
                       help='Output CSV file path (default: data_dir/insights/corner_clips_metadata.csv)')
    parser.add_argument('--split-by-visibility', action='store_true',
                       help='Split clips into visible/ and not_shown/ subdirectories')

    args = parser.parse_args()

    try:
        result_csv = extract_all_corners(args.data_dir, args.output, args.split_by_visibility)
        print(f"Corner clip extraction complete!")
        print(f"Results saved to: {result_csv}")

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()