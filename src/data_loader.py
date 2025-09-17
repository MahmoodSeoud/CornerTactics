#!/usr/bin/env python3
"""
SoccerNet Data Access
Load and access SoccerNet annotations and game data.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoccerNetDataLoader:
    """Access SoccerNet game data and annotations."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_annotations(self, game_path: str) -> Dict:
        """Load annotations for a game."""
        labels_file = self.data_dir / game_path / "Labels-v3.json"

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

        # Labels-v3.json has different structure: actions dict instead of annotations array
        for image_name, action_data in annotations.get('actions', {}).items():
            metadata = action_data.get('imageMetadata', {})
            if metadata.get('label') == 'Corner':
                game_time = metadata.get('gameTime', '')
                corners.append({
                    'gameTime': game_time,
                    'team': 'unknown',  # Team info not directly available in v3
                    'half': str(metadata.get('half', 1)),
                    'visibility': metadata.get('visibility', 'visible')
                })

        return corners