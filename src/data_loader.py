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
        
    def load_annotations(self, game_path: str, prefer_v2: bool = True) -> Dict:
        """Load annotations for a game."""
        if prefer_v2:
            # Check in data-old first for backward compatibility
            data_old_path = Path("data-old") / game_path / "Labels-v2.json"
            if data_old_path.exists():
                with open(data_old_path, 'r') as f:
                    return json.load(f)

            # Check in main data directory for v2 labels
            v2_path = self.data_dir / game_path / "Labels-v2.json"
            if v2_path.exists():
                with open(v2_path, 'r') as f:
                    return json.load(f)

        # Try Labels-v3.json
        v3_path = self.data_dir / game_path / "Labels-v3.json"
        if v3_path.exists():
            with open(v3_path, 'r') as f:
                return json.load(f)

        # Return empty dict if no labels found
        return {}
    
    def list_games(self) -> List[str]:
        """List all games with video files."""
        games = []

        # Look specifically in the videos directory
        videos_dir = self.data_dir / "datasets" / "soccernet" / "videos"

        if not videos_dir.exists():
            logger.warning(f"Videos directory not found: {videos_dir}")
            return games

        for root, dirs, files in os.walk(videos_dir):
            # Look for video files (any quality)
            video_files = [f for f in files if f.endswith('.mkv') and ('720p' in f or '224p' in f)]
            if video_files:
                relative_path = Path(root).relative_to(self.data_dir)
                game_path_str = str(relative_path)

                # Check if labels exist (v2 or v3)
                has_labels = False
                if self.use_v2_labels:
                    # Check for Labels-v2.json in data-old (backward compatibility)
                    v2_data_old = Path("data-old") / relative_path / "Labels-v2.json"
                    # Check for Labels-v2.json in main data directory
                    v2_data = self.data_dir / relative_path / "Labels-v2.json"
                    if v2_data_old.exists() or v2_data.exists():
                        has_labels = True

                if not has_labels:
                    # Check for Labels-v3.json in current data
                    v3_path = self.data_dir / relative_path / "Labels-v3.json"
                    if v3_path.exists():
                        has_labels = True

                if has_labels:
                    games.append(game_path_str)

        return sorted(games)
    
    def get_corner_events(self, game_path: str) -> List[Dict]:
        """Extract corner kick events from game annotations."""
        corners = []

        # Load v2 labels (has ALL corners)
        v2_annotations = self.load_annotations(game_path, prefer_v2=True)
        if 'annotations' in v2_annotations:
            for annotation in v2_annotations.get('annotations', []):
                if annotation.get('label') == 'Corner':
                    game_time = annotation.get('gameTime', '')
                    half = 1
                    if game_time and ' - ' in game_time:
                        half = int(game_time.split(' - ')[0])

                    corners.append({
                        'gameTime': game_time,
                        'team': annotation.get('team', 'unknown'),
                        'half': half,
                        'visibility': annotation.get('visibility', 'unknown'),
                        'source': 'v2'
                    })

        # Load v3 labels (has corners with spatial annotations)
        v3_annotations = self.load_annotations(game_path, prefer_v2=False)
        if 'actions' in v3_annotations:
            for image_name, action_data in v3_annotations.get('actions', {}).items():
                metadata = action_data.get('imageMetadata', {})
                if metadata.get('label') == 'Corner':
                    game_time = metadata.get('gameTime', '')

                    # Check if this corner already exists from v2
                    existing = any(c['gameTime'] == game_time for c in corners)
                    if not existing:
                        corners.append({
                            'gameTime': game_time,
                            'team': 'unknown',
                            'half': int(metadata.get('half', 1)),
                            'visibility': metadata.get('visibility', 'visible'),
                            'source': 'v3'
                        })

        return corners