#!/usr/bin/env python3
"""
Corner Frame Pipeline
Processes all games to extract corner frames and generate metadata CSV.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict

from data_loader import SoccerNetDataLoader
from frame_extractor import CornerFrameExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerFramePipeline:
    """Pipeline to extract frames for all corners and generate metadata CSV."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data_loader = SoccerNetDataLoader(data_dir)
        self.output_dir = Path(data_dir) / "insights"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_corners(self) -> str:
        """Extract frames for all corners and generate metadata CSV.

        Returns:
            Path to the generated CSV file
        """
        csv_path = self.output_dir / "corner_frames_metadata.csv"

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['game_path', 'game_time', 'half', 'team', 'visibility', 'frame_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Get all games with videos
            games = self.data_loader.list_games()
            logger.info(f"Processing {len(games)} games for corner extraction")

            for game_path in games:
                try:
                    # Get corner events for this game
                    corners = self.data_loader.get_corner_events(game_path)

                    if not corners:
                        continue

                    # Create frame extractor for this game
                    extractor = CornerFrameExtractor(game_path, self.data_dir)

                    # Extract frame for each corner
                    for corner in corners:
                        frame_path = extractor.extract_corner_frame(
                            corner['gameTime'],
                            corner.get('team', 'unknown'),
                            corner['half']
                        )

                        # Write to CSV (even if extraction failed)
                        writer.writerow({
                            'game_path': game_path,
                            'game_time': corner['gameTime'],
                            'half': str(corner['half']),
                            'team': corner.get('team', 'unknown'),
                            'visibility': corner.get('visibility', 'unknown'),
                            'frame_path': frame_path or ''
                        })

                except Exception as e:
                    logger.error(f"Failed to process game {game_path}: {e}")
                    continue

        logger.info(f"Corner frame extraction complete. Metadata saved to: {csv_path}")
        return str(csv_path)