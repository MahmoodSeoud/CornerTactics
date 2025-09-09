#!/usr/bin/env python3
"""
Corner Tactics Pipeline: Extract corner video clips and tracklets
Prerequisites: Download data using src/download_soccernet.py
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data_loader import SoccerNetDataLoader
from src.corner_extractor import CornerKickExtractor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Extract corner clips and tracklets from all games')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--no-clips', action='store_true', help='Skip video extraction')
    parser.add_argument('--duration', type=int, default=30, help='Clip duration (seconds)')
    parser.add_argument('--before', type=int, default=10, help='Seconds before corner')
    parser.add_argument('--output', default='corners.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Step 1: Get all available games
    loader = SoccerNetDataLoader(args.data_dir)
    games = loader.list_games()
    
    if not games:
        logger.error("No games found. Download data first:")
        logger.error("  python src/download_soccernet.py --videos 720p --tracklets tracking --password YOUR_PASSWORD")
        return
    
    logger.info(f"Found {len(games)} games")
    
    # Step 2: Extract corner events from all games
    logger.info("\n=== EXTRACTING CORNER EVENTS ===")
    all_corners = []
    
    for i, game in enumerate(games, 1):
        logger.info(f"[{i}/{len(games)}] {game}")
        try:
            corners = loader.get_corner_events(game)
            if corners:
                # Add game info to each corner
                for corner in corners:
                    corner['game'] = game
                all_corners.extend(corners)
                logger.info(f"  → Found {len(corners)} corners")
            else:
                logger.info(f"  → No corners found")
        except Exception as e:
            logger.error(f"  → Failed: {e}")
    
    if not all_corners:
        logger.error("No corner events found in any games")
        return
    
    # Step 3: Extract video clips for corners (optional)
    if not args.no_clips:
        logger.info(f"\n=== EXTRACTING {len(all_corners)} CORNER CLIPS ===")
        total_clips = 0
        
        for i, corner in enumerate(all_corners, 1):
            logger.info(f"[{i}/{len(all_corners)}] {corner['game']} - {corner['gameTime']}")
            try:
                extractor = CornerKickExtractor(corner['game'], args.data_dir)
                # Extract single corner clip
                clip_path = extractor.extract_corner_clip(
                    corner['gameTime'], corner['team'], 
                    args.duration, args.before
                )
                if clip_path:
                    corner['clip_path'] = str(clip_path)
                    total_clips += 1
                    logger.info(f"  → Extracted clip: {clip_path.name}")
                else:
                    logger.info(f"  → No clip extracted")
            except Exception as e:
                logger.error(f"  → Failed: {e}")
        
        logger.info(f"\nTotal clips extracted: {total_clips}")
    
    # Step 4: Save results
    df = pd.DataFrame(all_corners)
    df.to_csv(args.output, index=False)
    
    logger.info("\n=== RESULTS ===")
    logger.info(f"Total corners: {len(df)}")
    logger.info(f"Total games: {len(games)}")
    logger.info(f"Saved to: {args.output}")
    
    # Show summary statistics
    logger.info(f"\nCorners by team:")
    for team, count in df['team'].value_counts().head(10).items():
        logger.info(f"  {team}: {count}")
    
    logger.info(f"\nCorners by half:")
    for half, count in df['half'].value_counts().items():
        logger.info(f"  Half {half}: {count}")


if __name__ == "__main__":
    main()