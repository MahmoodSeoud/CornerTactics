#!/usr/bin/env python3
"""
Extract corner kick information from SoccerNet data and optionally create video clips.

Carmack-style: Simple functions that do one thing well.
"""

import argparse
import csv
import subprocess
import sys
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict

# Constants
CORNER_CLIPS_DIR = "datasets/soccernet/corner_clips"
VIDEO_QUALITY = "720p"
FFMPEG_TIMEOUT = 60
CLIP_DURATION = 20
CLIP_START_OFFSET = -5

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_game_time(time_string: str) -> Optional[int]:
    """Parse SoccerNet time format to seconds. Returns None on error."""
    try:
        # Handle "1 - 05:30" (v2) or "05:30" (v3) formats
        time_part = time_string.split(' - ')[-1]
        minutes, seconds = map(int, time_part.split(':'))
        return max(0, minutes * 60 + seconds)
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
    path_parts = game_path.split('/')
    game_id = '_'.join(path_parts[-3:] if len(path_parts) >= 3 else path_parts)

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
        return base_dir / "visible"


def extract_clip_with_ffmpeg(video_path: Path, start_seconds: int, duration: int, output_path: Path) -> bool:
    """Run ffmpeg to extract a video clip. Returns True on success."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_seconds),
        '-i', str(video_path),
        '-t', str(duration),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"ffmpeg failed: {e}")
        return False


def load_labels(game_dir: Path) -> List[Dict]:
    """Load corner labels from SoccerNet Labels-v2.json or Labels-v3.json"""
    corners = []

    # Try v3 first, then v2
    for labels_file in ['Labels-v3.json', 'Labels-v2.json']:
        labels_path = game_dir / labels_file
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    data = json.load(f)

                # Extract annotations
                annotations = data.get('annotations', [])
                for ann in annotations:
                    label = ann.get('label', '')
                    if 'corner' in label.lower():
                        corners.append({
                            'gameTime': ann.get('gameTime', ''),
                            'label': label,
                            'team': ann.get('team', 'unknown'),
                            'visibility': ann.get('visibility', 'visible'),
                            'half': int(ann.get('gameTime', '1 - 00:00').split(' - ')[0])
                        })

                if corners:
                    logger.debug(f"Loaded {len(corners)} corners from {labels_file}")
                    break
            except Exception as e:
                logger.error(f"Error loading {labels_path}: {e}")

    return corners


def list_games(data_dir: Path) -> List[str]:
    """List all game directories in SoccerNet format."""
    games = []

    # SoccerNet structure: league/season/game
    for league_dir in data_dir.iterdir():
        if not league_dir.is_dir() or league_dir.name.startswith('.'):
            continue

        for season_dir in league_dir.iterdir():
            if not season_dir.is_dir() or season_dir.name.startswith('.'):
                continue

            for game_dir in season_dir.iterdir():
                if not game_dir.is_dir() or game_dir.name.startswith('.'):
                    continue

                # Check if has labels
                if (game_dir / 'Labels-v2.json').exists() or (game_dir / 'Labels-v3.json').exists():
                    relative_path = str(game_dir.relative_to(data_dir))
                    games.append(relative_path)

    return sorted(games)


def extract_corner_clip(data_dir: Path, game_path: str, corner: Dict,
                       split_by_visibility: bool) -> Optional[str]:
    """Extract one corner video clip. Returns output path or None."""

    corner_timestamp = parse_game_time(corner['gameTime'])
    if corner_timestamp is None:
        return None

    clip_start_time = max(0, corner_timestamp + CLIP_START_OFFSET)

    video_path = find_video_file(str(data_dir), game_path, corner['half'])
    if video_path is None:
        return None

    output_dir = get_output_directory(str(data_dir), corner['visibility'], split_by_visibility)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = create_clip_filename(game_path, corner['half'], corner_timestamp, corner['team'])
    output_path = output_dir / filename

    success = extract_clip_with_ffmpeg(video_path, clip_start_time, CLIP_DURATION, output_path)

    if success:
        logger.info(f"Extracted clip: {filename}")
        return str(output_path)

    return None


def process_game(data_dir: Path, game_path: str, split_by_visibility: bool,
                extract_videos: bool) -> List[Dict]:
    """Process all corners in one game. Returns list of corner metadata."""

    game_dir = data_dir / game_path
    corners = load_labels(game_dir)

    if not corners:
        return []

    logger.info(f"Processing {len(corners)} corners in {game_path}")

    corner_data = []
    for corner in corners:
        clip_path = None

        if extract_videos:
            clip_path = extract_corner_clip(data_dir, game_path, corner, split_by_visibility)

        corner_data.append({
            'game_path': game_path,
            'game_time': corner['gameTime'],
            'half': str(corner['half']),
            'team': corner['team'],
            'label': corner['label'],
            'visibility': corner['visibility'],
            'clip_path': clip_path or '',
            'video_available': find_video_file(str(data_dir), game_path, corner['half']) is not None
        })

    return corner_data


def extract_all_corners(data_dir: str, output_csv: str, split_by_visibility: bool,
                       extract_videos: bool) -> str:
    """Extract corner metadata and optionally video clips from all SoccerNet games."""

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    games = list_games(data_path)

    if not games:
        logger.warning("No games found with labels")
        return ""

    logger.info(f"Found {len(games)} games with labels")

    # Setup output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process all games
    all_corners = []

    for i, game_path in enumerate(games, 1):
        logger.info(f"Game {i}/{len(games)}: {game_path}")

        try:
            corners = process_game(data_path, game_path, split_by_visibility, extract_videos)
            all_corners.extend(corners)
        except Exception as e:
            logger.error(f"Failed processing {game_path}: {e}")

    # Write CSV
    if all_corners:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['game_path', 'game_time', 'half', 'team', 'label',
                         'visibility', 'video_available', 'clip_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_corners)

        logger.info(f"Extracted {len(all_corners)} corner records")
        logger.info(f"Results saved to: {output_path}")

        # Summary
        with_video = sum(1 for c in all_corners if c['video_available'])
        extracted = sum(1 for c in all_corners if c['clip_path'])
        logger.info(f"  Corners with video available: {with_video}/{len(all_corners)}")
        if extract_videos:
            logger.info(f"  Video clips extracted: {extracted}/{with_video}")
    else:
        logger.warning("No corners found")

    return str(output_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract corner kick data from SoccerNet',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', required=True,
                       help='Root data directory containing SoccerNet data')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    parser.add_argument('--split-by-visibility', action='store_true',
                       help='Split clips into visible/ and not_shown/ subdirectories')
    parser.add_argument('--no-extract-videos', action='store_true',
                       help='Only create CSV metadata, skip video extraction')

    args = parser.parse_args()

    try:
        extract_videos = not args.no_extract_videos
        result_csv = extract_all_corners(
            args.data_dir,
            args.output,
            args.split_by_visibility,
            extract_videos
        )

        if result_csv:
            print(f"\n✓ Corner extraction complete!")
            print(f"Results saved to: {result_csv}")
        else:
            print(f"\n⊘ No corners found")
            sys.exit(1)

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
