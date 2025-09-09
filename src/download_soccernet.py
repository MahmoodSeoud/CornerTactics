#!/usr/bin/env python3
"""
SoccerNet Download Script
Direct interface for downloading broadcast videos and tracking data.
"""

import argparse
import sys
import logging
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_broadcast_videos(data_dir: str, quality: str, splits: list, password: str):
    """Download broadcast videos in specified quality."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    downloader = SoccerNetDownloader(LocalDirectory=str(data_path))
    if password:
        downloader.password = password
    
    files = [f"1_{quality}.mkv", f"2_{quality}.mkv"]
    logger.info(f"Downloading {quality} broadcast videos for splits: {splits}")
    
    try:
        downloader.downloadGames(files=files, split=splits)
        logger.info(f"Successfully downloaded {quality} videos")
    except Exception as e:
        if "google-analytics.com" in str(e):
            logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
        else:
            logger.error(f"Failed to download {quality} videos: {e}")
            raise


def download_tracklets(data_dir: str, task: str, splits: list):
    """Download tracking data."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    downloader = SoccerNetDownloader(LocalDirectory=str(data_path))
    
    logger.info(f"Downloading tracklets '{task}' for splits: {splits}")
    try:
        downloader.downloadDataTask(task=task, split=splits)
        logger.info(f"Successfully downloaded tracklets '{task}'")
    except Exception as e:
        if "google-analytics.com" in str(e):
            logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
        else:
            logger.error(f"Failed to download tracklets '{task}': {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Download SoccerNet broadcast videos and tracking data')
    
    # Video downloads
    parser.add_argument('--videos', choices=['720p', '224p'], help='Download broadcast videos (720p or 224p)')
    parser.add_argument('--tracklets', choices=['tracking', 'tracking-2023'], help='Download tracking data')
    
    # Options
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test', 'challenge'], 
                       help='Splits to download (default: all)')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--password', required=False, 
                       help='Password for videos (required for video downloads)')
    
    args = parser.parse_args()
    
    if not args.videos and not args.tracklets:
        parser.print_help()
        sys.exit(1)
    
    # Check password requirement for videos
    if args.videos and not args.password:
        print("ERROR: --password required for video downloads")
        sys.exit(1)
    
    # Download videos
    if args.videos:
        print(f"Downloading {args.videos} broadcast videos...")
        download_broadcast_videos(args.data_dir, args.videos, args.splits, args.password)
    
    # Download tracklets
    if args.tracklets:
        print(f"Downloading {args.tracklets} tracklets...")
        download_tracklets(args.data_dir, args.tracklets, args.splits)
    
    print("Download complete!")


if __name__ == '__main__':
    main()