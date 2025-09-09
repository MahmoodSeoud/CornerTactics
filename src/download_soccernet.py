#!/usr/bin/env python3
"""
SoccerNet Download Script
Command-line interface for downloading SoccerNet data.
"""

import argparse
import sys
from data_loader import SoccerNetDataLoader


class SoccerNetDownloadScript:
    """Command-line script for downloading SoccerNet data."""
    
    def __init__(self, data_dir: str = "data", password: str = "s0cc3rn3t"):
        self.data_dir = data_dir
        self.password = password
        self.data_loader = SoccerNetDataLoader(data_dir, password)
    
    def download_all_labels(self, splits):
        """Download labels for all specified splits."""
        for split in splits:
            self.data_loader.download_annotations(split)
    
    def download_videos_for_games(self, game_paths):
        """Download videos for specified games."""
        for game_path in game_paths:
            self.data_loader.download_videos(game_path)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Download SoccerNet labels and videos')
    parser.add_argument('--labels', nargs='+', help='Download labels for specified splits (e.g., train test)')
    parser.add_argument('--videos', nargs='+', help='Download videos for specified game paths')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--password', default='s0cc3rn3t', help='SoccerNet password (default: s0cc3rn3t)')
    
    args = parser.parse_args()
    
    script = SoccerNetDownloadScript(args.data_dir, args.password)
    
    if args.labels:
        script.download_all_labels(args.labels)
    
    if args.videos:
        script.download_videos_for_games(args.videos)
    
    if not args.labels and not args.videos:
        parser.print_help()


if __name__ == '__main__':
    main()