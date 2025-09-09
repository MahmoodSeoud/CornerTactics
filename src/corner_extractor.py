#!/usr/bin/env python3
"""
Corner Kick Extractor
Extract corner kick video clips from match videos.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerKickExtractor:
    """Extract corner kick clips from match videos."""
    
    def __init__(self, game_path: str, data_dir: str = "data", output_dir: str = "corner_clips"):
        self.game_path = Path(data_dir) / game_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_corner_kicks(self) -> List[Dict]:
        """Get all corner kick events from annotations."""
        labels_file = self.game_path / "Labels-v2.json"
        
        with open(labels_file, 'r') as f:
            data = json.load(f)
            
        corners = []
        for annotation in data['annotations']:
            if annotation['label'] == 'Corner':
                game_time = annotation['gameTime']
                half, time_str = game_time.split(' - ')
                minutes, seconds = map(int, time_str.split(':'))
                
                corners.append({
                    'half': int(half),
                    'minutes': minutes,
                    'seconds': seconds,
                    'total_seconds': minutes * 60 + seconds,
                    'team': annotation['team'],
                    'game_time': game_time
                })
                
        return corners
    
    def extract_clip(self, corner: Dict, duration: int = 30, before: int = 10) -> str:
        """Extract a video clip for a corner kick."""
        video_file = self.game_path / f"{corner['half']}.mkv"
        
        if not video_file.exists():
            logger.warning(f"Video file {video_file.name} not found")
            return None
            
        start_time = max(0, corner['total_seconds'] - before)
        output_file = self.output_dir / f"corner_{corner['half']}H_{corner['minutes']:02d}m{corner['seconds']:02d}s_{corner['team']}.mp4"
        
        cmd = [
            'ffmpeg',
            '-i', str(video_file),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Extracted: {output_file.name}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract clip: {e}")
            return None
    
    def extract_all(self, duration: int = 30, before: int = 10) -> List[str]:
        """Extract all corner kick clips."""
        corners = self.get_corner_kicks()
        logger.info(f"Found {len(corners)} corner kicks")
        
        clips = []
        for corner in corners:
            clip = self.extract_clip(corner, duration, before)
            if clip:
                clips.append(clip)
                
        logger.info(f"Extracted {len(clips)} clips")
        return clips