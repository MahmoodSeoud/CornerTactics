#!/usr/bin/env python3
"""
SoccerNet Download Script for CornerTactics Project

Downloads essential data for corner kick analysis:
1. Labels-v3.json - Corner event timestamps
2. Broadcast videos (720p) - For extracting 30s corner clips
3. Tracking data - Ground truth for 12 corner sequences
"""

import argparse
import sys
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader


def download_corner_labels(data_dir: str, splits: list):
    """Download Labels-v3.json containing corner event annotations."""
    downloader = SoccerNetDownloader(LocalDirectory=data_dir)

    print(f"Downloading corner labels for splits: {splits}")
    downloader.downloadGames(files=["Labels-v3.json"], split=splits)
    print("✓ Corner labels downloaded!")


def download_broadcast_videos(data_dir: str, splits: list, password: str):
    """Download 720p broadcast videos for corner clip extraction."""
    downloader = SoccerNetDownloader(LocalDirectory=data_dir)
    downloader.password = password

    print(f"Downloading 720p videos for splits: {splits}")
    print("Warning: This requires ~100GB per split!")

    # Download both halves of each game
    files = ["1_720p.mkv", "2_720p.mkv"]
    downloader.downloadGames(files=files, split=splits)
    print("✓ Broadcast videos downloaded!")


def download_tracking_data(data_dir: str, splits: list):
    """Download SNMOT tracking data (includes 12 corner sequences)."""
    downloader = SoccerNetDownloader(LocalDirectory=data_dir)

    print(f"Downloading tracking data for splits: {splits}")
    downloader.downloadDataTask(task="tracking", split=splits)
    print("✓ Tracking data downloaded!")


def main():
    parser = argparse.ArgumentParser(
        description='Download SoccerNet data for CornerTactics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Download corner labels only (lightweight, ~100MB)
  python src/download_soccernet.py --labels

  # Step 2: Download tracking data for validation (moderate, ~5GB)
  python src/download_soccernet.py --tracking

  # Step 3: Download broadcast videos (heavy, ~300GB, requires password)
  python src/download_soccernet.py --videos --password YOUR_PASSWORD

  # Or download everything at once:
  python src/download_soccernet.py --all --password YOUR_PASSWORD
        """
    )

    # Download options
    parser.add_argument('--labels', action='store_true',
                       help='Download Labels-v3.json (corner timestamps)')
    parser.add_argument('--videos', action='store_true',
                       help='Download 720p broadcast videos (requires password)')
    parser.add_argument('--tracking', action='store_true',
                       help='Download SNMOT tracking data')
    parser.add_argument('--all', action='store_true',
                       help='Download all data (labels + videos + tracking)')

    # Configuration
    parser.add_argument('--password', type=str,
                       help='SoccerNet password (required for videos)')
    parser.add_argument('--splits', nargs='+',
                       default=['train', 'valid', 'test'],
                       help='Data splits to download (default: train valid test)')
    parser.add_argument('--data-dir', default='./data/datasets/soccernet',
                       help='Base directory for SoccerNet data')

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        args.labels = True
        args.videos = True
        args.tracking = True

    # Check if any option selected
    if not any([args.labels, args.videos, args.tracking]):
        parser.print_help()
        sys.exit(1)

    # Validate password requirement
    if args.videos and not args.password:
        print("ERROR: --password required for downloading videos")
        print("Get your password at: https://www.soccer-net.org/")
        sys.exit(1)

    # Create data directory
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Execute downloads in order of size/importance
    if args.labels:
        download_corner_labels(args.data_dir, args.splits)

    if args.tracking:
        download_tracking_data(args.data_dir, args.splits)

    if args.videos:
        download_broadcast_videos(args.data_dir, args.splits, args.password)

    print("\n✓ All downloads complete!")
    print(f"Data saved to: {args.data_dir}")


if __name__ == '__main__':
    main()