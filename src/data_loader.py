#!/usr/bin/env python3
"""
SoccerNet Data Loader
Simple downloader for broadcast videos and tracking data.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict
from SoccerNet.Downloader import SoccerNetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoccerNetDataLoader:
    """Download SoccerNet broadcast videos and tracking data."""
    
    def __init__(self, data_dir: str = "data", password: str = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.downloader = SoccerNetDownloader(LocalDirectory=str(self.data_dir))
        if password:
            self.downloader.password = password
        
    def download_broadcast_videos(self, quality: str = "720p", splits: list = ["train", "valid", "test", "challenge"]):
        """
        Download broadcast videos in specified quality.
        Quality options: 720p, 224p
        """
        files = [f"1_{quality}.mkv", f"2_{quality}.mkv"]
        logger.info(f"Downloading {quality} broadcast videos for splits: {splits}")
        
        try:
            self.downloader.downloadGames(files=files, split=splits)
            logger.info(f"Successfully downloaded {quality} videos")
        except Exception as e:
            if "google-analytics.com" in str(e):
                logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
            else:
                logger.error(f"Failed to download {quality} videos: {e}")
                raise
    
    def download_tracklets(self, task: str = "tracking", splits: list = ["train", "test", "challenge"]):
        """
        Download tracking data.
        Available tasks: tracking, tracking-2023
        """
        logger.info(f"Downloading tracklets '{task}' for splits: {splits}")
        try:
            self.downloader.downloadDataTask(task=task, split=splits)
            logger.info(f"Successfully downloaded tracklets '{task}'")
        except Exception as e:
            if "google-analytics.com" in str(e):
                logger.warning(f"Analytics connection failed, but download may have succeeded: {e}")
            else:
                logger.error(f"Failed to download tracklets '{task}': {e}")
                raise
    
    def load_annotations(self, game_path: str) -> Dict:
        """Load annotations for a game."""
        labels_file = self.data_dir / game_path / "Labels-v2.json"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Annotations not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            return json.load(f)
    
    def list_games(self) -> List[str]:
        """List all games with video files."""
        games = []
        
        for root, dirs, files in os.walk(self.data_dir):
            # Look for video files (any quality)
            video_files = [f for f in files if f.endswith('.mkv') and ('720p' in f or '224p' in f)]
            if video_files:
                relative_path = Path(root).relative_to(self.data_dir)
                games.append(str(relative_path))
        
        return sorted(games)
    
    def get_corner_events(self, game_path: str) -> List[Dict]:
        """Extract corner kick events from game annotations."""
        annotations = self.load_annotations(game_path)
        corners = []
        
        for annotation in annotations.get('annotations', []):
            if annotation.get('label') == 'Corner':
                corners.append({
                    'gameTime': annotation.get('gameTime'),
                    'team': annotation.get('team'),
                    'half': annotation.get('gameTime', '').split(' - ')[0] if ' - ' in annotation.get('gameTime', '') else '1',
                    'visibility': annotation.get('visibility', 'visible')
                })
        
        return corners