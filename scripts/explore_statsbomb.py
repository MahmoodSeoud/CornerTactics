#!/usr/bin/env python3
"""
Explore StatsBomb Data
Discover available competitions and extract corner kick events with outcomes.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.statsbomb_loader import StatsBombCornerLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Explore StatsBomb data and extract corner kick samples."""

    logger.info("=" * 70)
    logger.info("StatsBomb Data Exploration")
    logger.info("=" * 70)

    # Initialize loader
    loader = StatsBombCornerLoader(output_dir="data/statsbomb")

    # Step 1: Get available competitions
    logger.info("\n[1/4] Fetching available competitions...")
    try:
        competitions = loader.get_available_competitions()

        logger.info(f"\nFound {len(competitions)} competitions in StatsBomb open data")
        logger.info("\nSample competitions:")
        logger.info(competitions.head(10).to_string())

        # Save to CSV
        loader.save_dataset(competitions, "available_competitions.csv")

    except Exception as e:
        logger.error(f"Error fetching competitions: {e}")
        return 1

    # Step 2: Select a sample competition for corner analysis
    # Let's use a popular competition (e.g., Premier League 2003/2004 if available)
    logger.info("\n[2/4] Selecting sample competition for analysis...")

    # Find a good sample competition
    sample_comp = None

    # Try to find Premier League
    pl_comps = competitions[
        competitions['competition_name'].str.contains('Premier League', case=False, na=False)
    ]
    if not pl_comps.empty:
        sample_comp = pl_comps.iloc[0]
        logger.info(f"Selected: {sample_comp['competition_name']} {sample_comp['season_name']}")
    else:
        # Use first available competition
        sample_comp = competitions.iloc[0]
        logger.info(f"Selected: {sample_comp['competition_name']} {sample_comp['season_name']}")

    # Step 3: Fetch events and filter corners
    logger.info(f"\n[3/4] Fetching corner kick events...")
    try:
        corner_dataset = loader.build_corner_dataset(
            country=sample_comp['country_name'],
            division=sample_comp['competition_name'],
            season=sample_comp['season_name'],
            gender=sample_comp['competition_gender']
        )

        # Display statistics
        logger.info("\n" + "=" * 70)
        logger.info("CORNER KICK STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total corner kicks: {len(corner_dataset)}")

        if len(corner_dataset) > 0:
            # Coordinate statistics
            logger.info(f"\nCorner coordinates (x,y):")
            logger.info(f"  X range: {corner_dataset['corner_x'].min():.1f} - {corner_dataset['corner_x'].max():.1f}")
            logger.info(f"  Y range: {corner_dataset['corner_y'].min():.1f} - {corner_dataset['corner_y'].max():.1f}")

            # End location statistics
            valid_ends = corner_dataset[corner_dataset['end_x'].notna()]
            if len(valid_ends) > 0:
                logger.info(f"\nTarget location (end_x, end_y):")
                logger.info(f"  X range: {valid_ends['end_x'].min():.1f} - {valid_ends['end_x'].max():.1f}")
                logger.info(f"  Y range: {valid_ends['end_y'].min():.1f} - {valid_ends['end_y'].max():.1f}")

            # Outcome distribution
            if 'outcome_type' in corner_dataset.columns:
                logger.info("\nOutcome type distribution:")
                outcome_counts = corner_dataset['outcome_type'].value_counts()
                for outcome, count in outcome_counts.items():
                    pct = (count / len(corner_dataset)) * 100
                    logger.info(f"  {outcome}: {count} ({pct:.1f}%)")

            # Shot outcome distribution (for corners leading to shots)
            if 'shot_outcome' in corner_dataset.columns:
                shots = corner_dataset[corner_dataset['shot_outcome'].notna()]
                if len(shots) > 0:
                    logger.info(f"\nShot outcomes ({len(shots)} shots):")
                    shot_counts = shots['shot_outcome'].value_counts()
                    for outcome, count in shot_counts.items():
                        pct = (count / len(shots)) * 100
                        logger.info(f"  {outcome}: {count} ({pct:.1f}%)")

            # Same team outcome
            if 'same_team_outcome' in corner_dataset.columns:
                same_team = corner_dataset['same_team_outcome'].sum()
                total_with_outcome = corner_dataset['same_team_outcome'].notna().sum()
                if total_with_outcome > 0:
                    pct = (same_team / total_with_outcome) * 100
                    logger.info(f"\nOutcomes by attacking team: {same_team}/{total_with_outcome} ({pct:.1f}%)")

            # Display sample rows
            logger.info("\n" + "=" * 70)
            logger.info("SAMPLE CORNER EVENTS (first 5)")
            logger.info("=" * 70)

            # Select key columns for display
            display_cols = [
                'team', 'minute', 'corner_x', 'corner_y', 'end_x', 'end_y',
                'outcome_type', 'shot_outcome', 'same_team_outcome'
            ]
            available_cols = [col for col in display_cols if col in corner_dataset.columns]

            logger.info("\n" + corner_dataset[available_cols].head().to_string())

        # Step 4: Save sample dataset
        logger.info(f"\n[4/4] Saving sample corner dataset...")
        loader.save_dataset(corner_dataset, "sample_corner_events.csv")

        logger.info("\n" + "=" * 70)
        logger.info("EXPLORATION COMPLETE")
        logger.info("=" * 70)
        logger.info("\nOutput files:")
        logger.info("  - data/statsbomb/available_competitions.csv")
        logger.info("  - data/statsbomb/sample_corner_events.csv")
        logger.info("\nNext steps:")
        logger.info("  1. Review sample data structure")
        logger.info("  2. Select competitions for full corner dataset extraction")
        logger.info("  3. Build training dataset with multiple competitions")
        logger.info("  4. Engineer geometric features (distances, angles)")
        logger.info("  5. Train outcome prediction model")

        return 0

    except Exception as e:
        logger.error(f"Error processing corner events: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
