#!/usr/bin/env python3
"""
Corner Kick Analyzer
Analyze corner kick patterns and outcomes.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


class CornerKickAnalyzer:
    """Analyze corner kick events and outcomes."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def analyze_game(self, game_path: str) -> pd.DataFrame:
        """Analyze corner kicks in a game."""
        labels_file = self.data_dir / game_path / "Labels-v2.json"
        
        with open(labels_file, 'r') as f:
            data = json.load(f)
            
        corners = []
        for annotation in data['annotations']:
            if annotation['label'] == 'Corner':
                game_time = annotation['gameTime']
                half, time_str = game_time.split(' - ')
                minutes, seconds = map(int, time_str.split(':'))
                
                corners.append({
                    'game': game_path,
                    'half': int(half),
                    'minutes': minutes,
                    'seconds': seconds,
                    'team': annotation['team'],
                    'visibility': annotation['visibility']
                })
                
        return pd.DataFrame(corners)
    
    def label_outcomes(self, game_path: str, window: int = 15) -> List[Dict]:
        """Label corner kick outcomes based on subsequent events."""
        labels_file = self.data_dir / game_path / "Labels-v2.json"
        
        with open(labels_file, 'r') as f:
            data = json.load(f)
            
        annotations = data['annotations']
        corners = []
        
        for i, annotation in enumerate(annotations):
            if annotation['label'] == 'Corner':
                game_time = annotation['gameTime']
                half, time_str = game_time.split(' - ')
                minutes, seconds = map(int, time_str.split(':'))
                corner_time = minutes * 60 + seconds
                
                # Look for outcome in next events
                outcome = 'continued_play'
                for next_ann in annotations[i+1:]:
                    next_time = self._parse_time(next_ann['gameTime'])
                    if next_time[0] != int(half):
                        break
                    if next_time[1] > corner_time + window:
                        break
                        
                    if 'goal' in next_ann['label'].lower():
                        outcome = 'goal'
                        break
                    elif 'clearance' in next_ann['label'].lower():
                        outcome = 'cleared'
                        break
                        
                corners.append({
                    'game': game_path,
                    'time': game_time,
                    'team': annotation['team'],
                    'outcome': outcome
                })
                
        return corners
    
    def _parse_time(self, game_time: str) -> tuple:
        """Parse game time string."""
        half, time_str = game_time.split(' - ')
        minutes, seconds = map(int, time_str.split(':'))
        return int(half), minutes * 60 + seconds