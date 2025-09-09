#!/usr/bin/env python3
"""
Simple pipeline: Extract clips from all games → Analyze all corners
Prerequisites: Download data first using data_loader.py
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data_loader import SoccerNetDataLoader
from src.corner_extractor import CornerKickExtractor
from src.analyzer import CornerKickAnalyzer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Extract and analyze corner kicks from all games')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--no-clips', action='store_true', help='Skip video extraction')
    parser.add_argument('--duration', type=int, default=30, help='Clip duration (seconds)')
    parser.add_argument('--before', type=int, default=10, help='Seconds before corner')
    parser.add_argument('--output', default='results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Step 1: Get all available games
    loader = SoccerNetDataLoader(args.data_dir)
    games = loader.list_games()
    
    if not games:
        logger.error("No games found. Download data first:")
        logger.error("  from src.data_loader import SoccerNetDataLoader")
        logger.error("  loader = SoccerNetDataLoader('data/')")
        logger.error("  loader.download_annotations('train')")
        return
    
    logger.info(f"Found {len(games)} games")
    
    # Step 2: Extract clips from all games (optional)
    if not args.no_clips:
        logger.info("\n=== EXTRACTING CORNER CLIPS ===")
        total_clips = 0
        
        for i, game in enumerate(games, 1):
            logger.info(f"[{i}/{len(games)}] {game}")
            try:
                extractor = CornerKickExtractor(game, args.data_dir)
                clips = extractor.extract_all(args.duration, args.before)
                total_clips += len(clips)
                logger.info(f"  → Extracted {len(clips)} clips")
            except Exception as e:
                logger.error(f"  → Failed: {e}")
        
        logger.info(f"\nTotal clips extracted: {total_clips}")
    
    # Step 3: Analyze all games
    logger.info("\n=== ANALYZING CORNERS ===")
    analyzer = CornerKickAnalyzer(args.data_dir)
    
    all_corners = []
    for i, game in enumerate(games, 1):
        logger.info(f"[{i}/{len(games)}] {game}")
        try:
            df = analyzer.analyze_game(game)
            all_corners.append(df)
            logger.info(f"  → Found {len(df)} corners")
        except Exception as e:
            logger.error(f"  → Failed: {e}")
    
    # Step 4: Combine and save results
    if all_corners:
        combined_df = pd.concat(all_corners, ignore_index=True)
        combined_df.to_csv(args.output, index=False)
        
        logger.info("\n=== RESULTS ===")
        logger.info(f"Total corners: {len(combined_df)}")
        logger.info(f"Total games: {len(all_corners)}")
        logger.info(f"Saved to: {args.output}")
        
        # Show team distribution
        logger.info("\nTop teams by corners:")
        for team, count in combined_df['team'].value_counts().head(5).items():
            logger.info(f"  {team}: {count}")
    else:
        logger.error("No corners found")


if __name__ == "__main__":
    main()