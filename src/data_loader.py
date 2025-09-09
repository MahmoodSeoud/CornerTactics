#!/usr/bin/env python3
"""
SoccerNet Data Loader
Simple data loading utilities for SoccerNet dataset.
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
    """Load and download SoccerNet data."""
    
    def __init__(self, data_dir: str = "data", password: str = "s0cc3rn3t"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.downloader = SoccerNetDownloader(LocalDirectory=str(self.data_dir))
        self.downloader.password = password
        
    def download_annotations(self, split: str = 'train'):
        """
        Download all annotation files for a split.
        Note: Downloads ALL games in the split.
        """
        logger.info(f"Downloading annotations for split: {split}")
        self.downloader.downloadGames(files=['Labels-v2.json'], split=[split])
        
    def download_videos(self, game_path: str):
        """Download video files for a specific game."""
        logger.info(f"Downloading videos for: {game_path}")
        
        for video_file in ['1.mkv', '2.mkv']:
            try:
                self.downloader.downloadVideo(game=game_path, file=video_file)
                logger.info(f"Downloaded {video_file}")
            except Exception as e:
                logger.error(f"Failed to download {video_file}: {e}")
    
    def load_annotations(self, game_path: str) -> Dict:
        """Load annotations for a game."""
        labels_file = self.data_dir / game_path / "Labels-v2.json"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Annotations not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            return json.load(f)
    
    def list_games(self) -> List[str]:
        """List all games with annotations."""
        games = []
        
        for root, dirs, files in os.walk(self.data_dir):
            if "Labels-v2.json" in files:
                relative_path = Path(root).relative_to(self.data_dir)
                games.append(str(relative_path))
        
        return sorted(games)