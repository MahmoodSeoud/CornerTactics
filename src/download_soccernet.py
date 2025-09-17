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


def download_labels(data_dir: str, version: str, splits: list):
    """Download match labels/annotations."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(data_path))

    # If version is "both", download both v2 and v3
    if version == "both":
        files = ["Labels-v2.json", "Labels-v3.json"]
    elif version:
        files = [f"Labels-{version}.json"]
    else:
        files = ["Labels-v2.json"]  # Default to v2

    logger.info(f"Downloading labels {files} for splits: {splits}")

    try:
        downloader.downloadGames(files=files, split=splits)
        logger.info(f"Successfully downloaded labels")
    except Exception as e:
        if "google-analytics.com" in str(e):
            logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
        else:
            logger.error(f"Failed to download labels: {e}")
            raise


def download_frames(data_dir: str, version: str, splits: list):
    """Download v3 frames with bounding boxes."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    downloader = SoccerNetDownloader(LocalDirectory=str(data_path))
    
    files = [f"Frames-{version}.zip"] if version else ["Frames-v3.zip"]
    logger.info(f"Downloading frames {files} for splits: {splits}")
    
    try:
        downloader.downloadGames(files=files, split=splits, task="frames")
        logger.info(f"Successfully downloaded frames")
    except Exception as e:
        if "google-analytics.com" in str(e):
            logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
        else:
            logger.error(f"Failed to download frames: {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Download SoccerNet broadcast videos and tracking data')
    
    # Download options
    parser.add_argument('--videos', choices=['720p', '224p'], help='Download broadcast videos (720p or 224p)')
    parser.add_argument('--tracklets', choices=['tracking', 'tracking-2023'], help='Download tracking data')
    parser.add_argument('--labels', choices=['v3', 'v2', 'v1'], help='Download match labels/annotations')
    parser.add_argument('--frames', choices=['v3'], help='Download v3 frames with bounding boxes')
    
    # Options
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test', 'challenge'], 
                       help='Splits to download (default: all)')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--password', required=False, 
                       help='Password for videos (required for video downloads)')
    
    args = parser.parse_args()
    
    if not args.videos and not args.tracklets and not args.labels and not args.frames:
        parser.print_help()
        sys.exit(1)
    
    # Check password requirement for videos
    if args.videos and not args.password:
        print("ERROR: --password required for video downloads")
        sys.exit(1)
    
    # Download labels
    if args.labels:
        print(f"Downloading labels {args.labels}...")
        download_labels(args.data_dir, args.labels, args.splits)
    
    # Download frames (v3)
    if args.frames:
        print(f"Downloading frames {args.frames}...")
        download_frames(args.data_dir, args.frames, args.splits)
    
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