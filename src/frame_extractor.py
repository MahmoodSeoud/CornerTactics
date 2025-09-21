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

    def __init__(self, game_path: str, data_dir: str = "data", output_dir: str = None, split_by_visibility: bool = False):
        self.game_path = Path(data_dir) / game_path
        self.split_by_visibility = split_by_visibility

        if output_dir is None:
            # Store frames in soccernet data structure
            self.base_output_dir = Path(data_dir) / "datasets" / "soccernet" / "corner_frames"
        else:
            self.base_output_dir = Path(output_dir)

        if split_by_visibility:
            # Create subdirectories for each visibility type
            self.visible_dir = self.base_output_dir / "visible"
            self.not_shown_dir = self.base_output_dir / "not_shown"
            self.visible_dir.mkdir(parents=True, exist_ok=True)
            self.not_shown_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Single output directory
            self.output_dir = self.base_output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame(self, game_time: str, team: str, half: int, offset_seconds: int = 0) -> Optional[str]:
        """Extract a single frame at the corner kick moment.

        Args:
            game_time: Game time string (e.g., "1 - 05:30" or just "05:30")
            team: Team name
            half: Half number (1 or 2)
            offset_seconds: Seconds offset from corner moment (default 0 = exact moment)

        Returns:
            Path to extracted frame or None if extraction failed
        """
        # Parse time - handle both v2 format ("1 - 05:30") and v3 format ("05:30")
        try:
            if ' - ' in game_time:
                # v2 format: "1 - 05:30"
                _, time_str = game_time.split(' - ')
            else:
                # v3 format: "05:30" or other simple formats
                time_str = game_time

            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds + offset_seconds

            # Validate time makes sense
            if total_seconds < 0:
                logger.warning(f"Negative timestamp after offset: {total_seconds}s")
                total_seconds = 0

        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse game time '{game_time}': {e}")
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

    def extract_corner_frame(self, game_time: str, team: str, half: int = None, visibility: str = "unknown") -> Optional[str]:
        """Extract a frame for a corner kick event.

        Args:
            game_time: Game time string (e.g., "1 - 05:30" or "05:30")
            team: Team name
            half: Half number (required for v3 format, optional for v2 format)
            visibility: Visibility label ("visible", "not shown", or "unknown")

        Returns:
            Path to extracted frame or None
        """
        # For v2 format, extract half from game_time if not provided
        if half is None:
            if ' - ' in game_time:
                try:
                    half_str, _ = game_time.split(' - ')
                    half = int(half_str)
                except (ValueError, AttributeError):
                    logger.error(f"Could not determine half from game_time: {game_time}")
                    return None
            else:
                logger.error(f"Half number required for game_time format: {game_time}")
                return None

        # Validate half number
        if half not in [1, 2]:
            logger.error(f"Invalid half number: {half}, must be 1 or 2")
            return None

        return self.extract_frame_with_visibility(game_time, team, half, visibility, offset_seconds=0)

    def extract_frame_with_visibility(self, game_time: str, team: str, half: int, visibility: str, offset_seconds: int = 0) -> Optional[str]:
        """Extract a single frame with visibility-based directory placement."""
        # Parse time - handle both v2 format ("1 - 05:30") and v3 format ("05:30")
        try:
            if ' - ' in game_time:
                # v2 format: "1 - 05:30"
                _, time_str = game_time.split(' - ')
            else:
                # v3 format: "05:30" or other simple formats
                time_str = game_time

            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds + offset_seconds

            # Validate time makes sense
            if total_seconds < 0:
                logger.warning(f"Negative timestamp after offset: {total_seconds}s")
                total_seconds = 0

        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse game time '{game_time}': {e}")
            return None

        video_file = self.game_path / f"{half}_720p.mkv"

        if not video_file.exists():
            logger.warning(f"Video file {video_file.name} not found")
            return None

        # Create output filename with game identifier to avoid collisions
        game_parts = str(self.game_path).split('/')
        if len(game_parts) >= 3:
            # Use last 3 parts: league/season/match
            game_id = '_'.join(game_parts[-3:])
        else:
            game_id = '_'.join(game_parts)

        # Sanitize game_id and team name
        safe_game = game_id.replace(' ', '_').replace('/', '_').replace('-', '').replace(':', '')[:50]
        safe_team = team.replace(' ', '_').replace('/', '_')
        output_filename = f"{safe_game}_{half}H_{minutes:02d}m{seconds:02d}s_{safe_team}.jpg"

        # Determine output directory based on visibility
        if self.split_by_visibility:
            if visibility == 'visible':
                output_dir = self.visible_dir
            elif visibility == 'not shown':
                output_dir = self.not_shown_dir
            else:
                # Unknown visibility - put in visible by default
                output_dir = self.visible_dir
        else:
            output_dir = self.output_dir

        output_file = output_dir / output_filename

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
            logger.info(f"Extracted frame ({visibility}): {output_file.name}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract frame: {e.stderr.decode()}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            return None