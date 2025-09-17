#!/usr/bin/env python3
"""
Corner Frame Extractor
Extract single frames from corner kick moments.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerFrameExtractor:
    """Extract single frames at corner kick moments."""

    def __init__(self, game_path: str, data_dir: str = "data", output_dir: str = None):
        self.game_path = Path(data_dir) / game_path
        if output_dir is None:
            # Store frames in soccernet data structure
            self.output_dir = Path(data_dir) / "datasets" / "soccernet" / "soccernet_corner_frames"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame(self, game_time: str, team: str, half: int, offset_seconds: int = 0) -> Optional[str]:
        """Extract a single frame at the corner kick moment.

        Args:
            game_time: Game time string (e.g., "1 - 05:30")
            team: Team name
            half: Half number (1 or 2)
            offset_seconds: Seconds offset from corner moment (default 0 = exact moment)

        Returns:
            Path to extracted frame or None if extraction failed
        """
        # Parse time
        try:
            _, time_str = game_time.split(' - ')
            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds + offset_seconds
        except:
            logger.error(f"Failed to parse game time: {game_time}")
            return None

        video_file = self.game_path / f"{half}_720p.mkv"

        if not video_file.exists():
            logger.warning(f"Video file {video_file.name} not found")
            return None

        # Create output filename with game identifier to avoid collisions
        # Extract a game identifier from the game path
        game_parts = str(self.game_path).split('/')
        if len(game_parts) >= 3:
            # Use last 3 parts: league/season/match
            game_id = '_'.join(game_parts[-3:])
        else:
            game_id = '_'.join(game_parts)

        # Sanitize game_id and team name
        safe_game = game_id.replace(' ', '_').replace('/', '_').replace('-', '').replace(':', '')[:50]  # Limit length
        safe_team = team.replace(' ', '_').replace('/', '_')
        output_file = self.output_dir / f"{safe_game}_{half}H_{minutes:02d}m{seconds:02d}s_{safe_team}.jpg"

        # Extract single frame using ffmpeg
        cmd = [
            'ffmpeg',
            '-ss', str(total_seconds),
            '-i', str(video_file),
            '-frames:v', '1',
            '-q:v', '2',  # High quality JPEG
            '-y',
            str(output_file)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            logger.info(f"Extracted frame: {output_file.name}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract frame: {e.stderr.decode()}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            return None

    def extract_corner_frame(self, game_time: str, team: str, half: int = None) -> Optional[str]:
        """Extract a frame for a corner kick event.

        Args:
            game_time: Full game time string (e.g., "1 - 05:30")
            team: Team name
            half: Optional half number (will parse from game_time if not provided)

        Returns:
            Path to extracted frame or None
        """
        if half is None:
            try:
                half_str, _ = game_time.split(' - ')
                half = int(half_str)
            except:
                logger.error(f"Could not determine half from game_time: {game_time}")
                return None

        return self.extract_frame(game_time, team, half, offset_seconds=0)