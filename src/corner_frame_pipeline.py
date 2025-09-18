#!/usr/bin/env python3
"""
Corner Frame Pipeline
Processes all games to extract corner frames and generate metadata CSV.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from data_loader import SoccerNetDataLoader
from frame_extractor import CornerFrameExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CornerFramePipeline:
    """Pipeline to extract frames for all corners and generate metadata CSV."""

    def __init__(self, data_dir: str, output_csv: Optional[str] = None):
        """Initialize the pipeline.

        Args:
            data_dir: Root data directory containing SoccerNet data
            output_csv: Optional custom path for output CSV file
        """
        self.data_dir = data_dir
        self.data_loader = SoccerNetDataLoader(data_dir)
        self.output_dir = Path(data_dir) / "insights"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure output CSV path
        if output_csv:
            self.csv_path = Path(output_csv)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.csv_path = self.output_dir / "corner_frames_metadata.csv"

    def extract_all_corners(self) -> str:
        """Extract frames for all corners and generate metadata CSV.

        Returns:
            Path to the generated CSV file
        """
        # Get all games with videos
        games = self.data_loader.list_games()
        logger.info(f"Found {len(games)} games with videos and labels")

        if not games:
            logger.warning("No games found with both videos and labels")
            return str(self.csv_path)

        total_corners = 0
        successful_extractions = 0
        failed_extractions = 0

        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['game_path', 'game_time', 'half', 'team', 'visibility', 'frame_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for game_index, game_path in enumerate(games, 1):
                logger.info(f"Processing game {game_index}/{len(games)}: {game_path}")

                try:
                    # Get corner events for this game
                    corners = self.data_loader.get_corner_events(game_path)

                    if not corners:
                        logger.debug(f"No corners found in game: {game_path}")
                        continue

                    logger.info(f"Found {len(corners)} corners in game")
                    total_corners += len(corners)

                    # Create frame extractor for this game
                    extractor = CornerFrameExtractor(game_path, self.data_dir)

                    # Extract frame for each corner
                    for corner_index, corner in enumerate(corners, 1):
                        logger.debug(f"Extracting corner {corner_index}/{len(corners)}: {corner['gameTime']}")

                        frame_path = extractor.extract_corner_frame(
                            corner['gameTime'],
                            corner.get('team', 'unknown'),
                            corner['half']
                        )

                        if frame_path:
                            successful_extractions += 1
                        else:
                            failed_extractions += 1

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

        # Log final statistics
        logger.info(f"Corner frame extraction complete!")
        logger.info(f"Total corners processed: {total_corners}")
        logger.info(f"Successful extractions: {successful_extractions}")
        logger.info(f"Failed extractions: {failed_extractions}")
        logger.info(f"Success rate: {(successful_extractions/total_corners*100):.1f}%" if total_corners > 0 else "N/A")
        logger.info(f"Metadata saved to: {self.csv_path}")

        return str(self.csv_path)