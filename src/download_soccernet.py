#!/usr/bin/env python3
"""
SoccerNet Download Script
Simple interface for downloading broadcast videos and tracking data.
"""

import argparse
import sys
from data_loader import SoccerNetDataLoader


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
    
    loader = SoccerNetDataLoader(args.data_dir, args.password)
    
    # Download videos
    if args.videos:
        print(f"Downloading {args.videos} broadcast videos...")
        loader.download_broadcast_videos(args.videos, args.splits)
    
    # Download tracklets
    if args.tracklets:
        print(f"Downloading {args.tracklets} tracklets...")
        loader.download_tracklets(args.tracklets, args.splits)
    
    print("Download complete!")


if __name__ == '__main__':
    main()