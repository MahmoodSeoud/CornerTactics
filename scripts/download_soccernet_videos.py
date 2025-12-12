#!/usr/bin/env python3
"""Download SoccerNet 720p videos for all 550 games.

Requires NDA password from SoccerNet. Set via environment variable:
    export SOCCERNET_PASSWORD="your_password"

Or pass via command line:
    python download_soccernet_videos.py --password "your_password"
"""

import os
import sys
import argparse
import time
import logging
import ssl
import certifi
from pathlib import Path

# Fix SSL certificate verification
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Alternative: disable SSL verification (less secure but works)
ssl._create_default_https_context = ssl._create_unverified_context

from SoccerNet.Downloader import SoccerNetDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_soccernet.log')
    ]
)
logger = logging.getLogger(__name__)


def download_videos(
    local_dir: str,
    password: str,
    splits: list = None,
    files: list = None,
    max_retries: int = 5,
    retry_delay: int = 60
):
    """Download SoccerNet videos with retry logic.

    Args:
        local_dir: Directory to save videos
        password: NDA password for video access
        splits: Dataset splits to download (default: all)
        files: Video files to download (default: 720p both halves)
        max_retries: Maximum retry attempts per download
        retry_delay: Seconds to wait between retries
    """
    if splits is None:
        splits = ["train", "valid", "test", "challenge"]

    if files is None:
        files = ["1_720p.mkv", "2_720p.mkv"]

    logger.info(f"Downloading SoccerNet videos to: {local_dir}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Files: {files}")

    # Initialize downloader
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.password = password

    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")

            # Download games - this will skip already downloaded files
            downloader.downloadGames(files=files, split=splits)

            logger.info("Download completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for subsequent retries
                retry_delay *= 2
            else:
                logger.error("Max retries exceeded")
                return False

    return False


def check_download_status(local_dir: str):
    """Check how many videos have been downloaded."""
    video_dir = Path(local_dir)

    if not video_dir.exists():
        logger.info("Download directory does not exist yet")
        return 0, 0

    # Count .mkv files
    mkv_files = list(video_dir.rglob("*_720p.mkv"))

    # Count unique games (each game has 2 halves)
    game_dirs = set()
    for f in mkv_files:
        game_dirs.add(f.parent)

    logger.info(f"Found {len(mkv_files)} video files across {len(game_dirs)} games")
    return len(mkv_files), len(game_dirs)


def main():
    parser = argparse.ArgumentParser(description='Download SoccerNet 720p videos')
    parser.add_argument(
        '--local-dir',
        type=str,
        default='/home/mseo/CornerTactics/data/misc/soccernet/videos',
        help='Directory to save videos'
    )
    parser.add_argument(
        '--password',
        type=str,
        default=os.environ.get('SOCCERNET_PASSWORD', ''),
        help='NDA password (or set SOCCERNET_PASSWORD env var)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test', 'challenge'],
        help='Dataset splits to download'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check download status, do not download'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=10,
        help='Maximum retry attempts'
    )
    args = parser.parse_args()

    # Check current status
    num_files, num_games = check_download_status(args.local_dir)

    if args.check_only:
        print(f"\nCurrent status: {num_files} video files, {num_games} games")
        print(f"Expected: 1100 video files (550 games x 2 halves)")
        print(f"Missing: ~{1100 - num_files} video files")
        return

    # Validate password
    if not args.password:
        logger.error("No password provided!")
        logger.error("Set SOCCERNET_PASSWORD environment variable or use --password")
        logger.error("Get password by signing NDA at: https://www.soccer-net.org/data")
        sys.exit(1)

    # Create output directory
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    # Start download
    success = download_videos(
        local_dir=args.local_dir,
        password=args.password,
        splits=args.splits,
        max_retries=args.max_retries
    )

    # Final status
    num_files, num_games = check_download_status(args.local_dir)
    logger.info(f"Final status: {num_files} video files, {num_games} games")

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
