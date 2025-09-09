#!/usr/bin/env python3
"""
Main pipeline for corner kick analysis.
Prerequisites: Data must be downloaded first using data_loader.py
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data_loader import SoccerNetDataLoader
from src.corner_extractor import CornerKickExtractor
from src.analyzer import CornerKickAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(game_path: str, data_dir: str = "data", 
                 extract_clips: bool = True, clip_duration: int = 30,
                 clip_before: int = 10):
    """
    Run complete corner kick analysis pipeline.
    
    Args:
        game_path: Path to game (e.g., 'england_epl/2015-2016/...')
        data_dir: Directory containing SoccerNet data
        extract_clips: Whether to extract video clips
        clip_duration: Total clip duration in seconds
        clip_before: Seconds before corner kick to include
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Step 1: Verify data exists
    logger.info(f"Processing game: {game_path}")
    
    loader = SoccerNetDataLoader(data_dir)
    try:
        annotations = loader.load_annotations(game_path)
        logger.info(f"Loaded annotations: {len(annotations['annotations'])} events")
    except FileNotFoundError:
        logger.error(f"Game not found. Please download it first.")
        logger.info("Usage: loader.download_videos(game_path)")
        return None
    
    # Step 2: Extract corner kick clips (optional)
    if extract_clips:
        logger.info("Extracting corner kick clips...")
        extractor = CornerKickExtractor(game_path, data_dir)
        clips = extractor.extract_all(duration=clip_duration, before=clip_before)
        results['clips'] = clips
        logger.info(f"Extracted {len(clips)} corner kick clips")
    
    # Step 3: Analyze corner kicks
    logger.info("Analyzing corner kicks...")
    analyzer = CornerKickAnalyzer(data_dir)
    
    # Get corner kick statistics
    df = analyzer.analyze_game(game_path)
    results['statistics'] = df
    
    # Label outcomes
    outcomes = analyzer.label_outcomes(game_path)
    results['outcomes'] = outcomes
    
    # Summary statistics
    if not df.empty:
        summary = {
            'total_corners': len(df),
            'corners_by_half': df['half'].value_counts().to_dict(),
            'corners_by_team': df['team'].value_counts().to_dict(),
            'visible_corners': (df['visibility'] == 'visible').sum()
        }
        results['summary'] = summary
        
        logger.info(f"Total corners: {summary['total_corners']}")
        logger.info(f"By team: {summary['corners_by_team']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Corner kick analysis pipeline')
    parser.add_argument('game', nargs='?', help='Game path (optional, will list games if not provided)')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--no-clips', action='store_true', help='Skip video clip extraction')
    parser.add_argument('--duration', type=int, default=30, help='Clip duration (seconds)')
    parser.add_argument('--before', type=int, default=10, help='Seconds before corner')
    parser.add_argument('--list', action='store_true', help='List available games')
    parser.add_argument('--output', help='Save results to CSV file')
    
    args = parser.parse_args()
    
    loader = SoccerNetDataLoader(args.data_dir)
    
    # List available games
    if args.list or not args.game:
        games = loader.list_games()
        print(f"\nAvailable games ({len(games)} total):")
        for game in games[:10]:  # Show first 10
            print(f"  {game}")
        if len(games) > 10:
            print(f"  ... and {len(games) - 10} more")
        
        if not args.game:
            print("\nUsage: python main.py <game_path>")
            print("Example: python main.py 'england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom'")
        return
    
    # Run pipeline
    results = run_pipeline(
        args.game,
        args.data_dir,
        extract_clips=not args.no_clips,
        clip_duration=args.duration,
        clip_before=args.before
    )
    
    if results:
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Total corners: {summary['total_corners']}")
            print(f"By half: {summary['corners_by_half']}")
            print(f"By team: {summary['corners_by_team']}")
        
        if 'clips' in results:
            print(f"\nExtracted clips: {len(results['clips'])}")
            for clip in results['clips']:
                print(f"  - {Path(clip).name}")
        
        # Save results if requested
        if args.output and 'statistics' in results:
            results['statistics'].to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()