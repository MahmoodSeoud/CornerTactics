#!/usr/bin/env python3
"""
Create Unified Corner Dataset
Combines StatsBomb 360, SkillCorner, and SoccerNet corner data into unified format
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_statsbomb_corners(data_dir):
    """Load StatsBomb corners with outcomes and 360 positions"""
    print("Loading StatsBomb corners...")

    file_path = data_dir / "datasets" / "statsbomb" / "corners_360_with_outcomes.csv"
    if not file_path.exists():
        print(f"  Warning: {file_path} not found, trying without outcomes...")
        file_path = data_dir / "datasets" / "statsbomb" / "corners_360.csv"

    df = pd.read_csv(file_path)

    # Standardize columns
    df['source'] = 'statsbomb'
    df['corner_id'] = 'sb_' + df['match_id'].astype(str) + '_' + df.index.astype(str)
    df['has_player_positions'] = df['attacking_positions'].notna()
    df['has_tracking'] = False  # StatsBomb has freeze frames, not continuous tracking
    df['has_outcome'] = 'outcome_category' in df.columns

    print(f"  Loaded {len(df)} StatsBomb corners")
    print(f"    With player positions: {df['has_player_positions'].sum()}")
    if 'outcome_category' in df.columns:
        print(f"    With outcomes: {df['has_outcome'].sum()}")

    return df

def load_skillcorner_corners(data_dir):
    """Load SkillCorner corners with continuous tracking"""
    print("\nLoading SkillCorner corners...")

    file_path = data_dir / "datasets" / "skillcorner" / "skillcorner_corners.csv"
    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Standardize columns
    df['source'] = 'skillcorner'
    df['corner_id'] = 'sc_' + df['match_id'].astype(str) + '_' + df['event_id'].astype(str)
    df['has_player_positions'] = False  # Will need to extract from tracking
    df['has_tracking'] = df['has_tracking'] if 'has_tracking' in df.columns else False
    df['has_outcome'] = False  # Will need to label outcomes separately

    print(f"  Loaded {len(df)} SkillCorner corners")
    print(f"    With tracking data: {df['has_tracking'].sum()}")

    return df

def load_soccernet_metadata(data_dir):
    """Load SoccerNet corner metadata from CSV"""
    print("\nLoading SoccerNet corners...")

    soccernet_csv = data_dir / "datasets" / "soccernet" / "soccernet_corners.csv"
    if not soccernet_csv.exists():
        print(f"  Warning: {soccernet_csv} not found")
        print(f"  Run extract_soccernet_corners.py first")
        return pd.DataFrame()

    df = pd.read_csv(soccernet_csv)

    # Standardize columns
    df['source'] = 'soccernet'
    df['corner_id'] = 'sn_' + df['game_path'].str.replace('/', '_') + '_' + df['game_time'].str.replace(':', '').str.replace(' - ', '_')
    df['has_video'] = df['video_available'] if 'video_available' in df.columns else False
    df['has_player_positions'] = False
    df['has_tracking'] = False
    df['has_outcome'] = False

    print(f"  Loaded {len(df)} SoccerNet corners")
    if 'video_available' in df.columns:
        print(f"    With video available: {df['video_available'].sum()}")
    if 'clip_path' in df.columns:
        clips_extracted = df['clip_path'].notna().sum()
        print(f"    With clips extracted: {clips_extracted}")

    return df

def create_unified_schema(statsbomb_df, skillcorner_df, soccernet_df):
    """Create unified corner dataset with standardized schema"""
    print("\nCreating unified dataset...")

    # Define unified schema
    unified_corners = []

    # Process StatsBomb corners
    for _, corner in statsbomb_df.iterrows():
        unified_corner = {
            'corner_id': corner['corner_id'],
            'source': 'statsbomb',

            # Match info
            'match_id': corner['match_id'],
            'competition': corner.get('competition', None),
            'season': corner.get('season', None),
            'match_date': corner.get('match_date', None),
            'home_team': corner.get('home_team', None),
            'away_team': corner.get('away_team', None),

            # Timing
            'period': corner.get('period', None),
            'minute': corner.get('minute', None),
            'second': corner.get('second', None),

            # Team/Player
            'attacking_team': corner.get('team', None),
            'player_name': corner.get('player', None),

            # Location (StatsBomb coordinates: 120x80)
            'corner_x': corner.get('corner_x', None),
            'corner_y': corner.get('corner_y', None),
            'target_x': corner.get('end_x', None),
            'target_y': corner.get('end_y', None),

            # Outcome
            'outcome_category': corner.get('outcome_category', None),
            'outcome_type': corner.get('outcome_type', None),
            'goal_scored': corner.get('goal_scored', False),
            'time_to_outcome': corner.get('time_to_outcome', None),

            # Data availability
            'has_player_positions': corner['has_player_positions'],
            'has_tracking': False,
            'has_video': False,
            'player_positions_json': corner.get('attacking_positions', None),
            'tracking_file': None,
            'video_path': None
        }

        unified_corners.append(unified_corner)

    # Process SkillCorner corners
    for _, corner in skillcorner_df.iterrows():
        unified_corner = {
            'corner_id': corner['corner_id'],
            'source': 'skillcorner',

            # Match info
            'match_id': corner['match_id'],
            'competition': corner.get('competition', 'A-League'),
            'season': corner.get('season', None),
            'match_date': corner.get('match_date', None),
            'home_team': corner.get('home_team', None),
            'away_team': corner.get('away_team', None),

            # Timing
            'period': corner.get('period', None),
            'minute': corner.get('minute_start', None),
            'second': corner.get('second_start', None),

            # Team/Player
            'attacking_team': corner.get('team_shortname', None),
            'player_name': corner.get('player_name', None),

            # Location (SkillCorner coordinates: normalized)
            'corner_x': corner.get('x_start', None),
            'corner_y': corner.get('y_start', None),
            'target_x': corner.get('x_end', None),
            'target_y': corner.get('y_end', None),

            # Outcome (to be labeled later)
            'outcome_category': None,
            'outcome_type': None,
            'goal_scored': False,
            'time_to_outcome': None,

            # Data availability
            'has_player_positions': False,  # Will extract from tracking
            'has_tracking': corner['has_tracking'],
            'has_video': False,
            'player_positions_json': None,
            'tracking_file': corner.get('tracking_file', None),
            'video_path': None
        }

        unified_corners.append(unified_corner)

    # Process SoccerNet corners
    for _, corner in soccernet_df.iterrows():
        # Parse game path: league/season/match
        game_parts = corner['game_path'].split('/')
        competition = game_parts[0] if len(game_parts) > 0 else None
        season = game_parts[1] if len(game_parts) > 1 else None

        # Parse time: "1 - 05:30" -> half=1, minute=5, second=30
        game_time = corner.get('game_time', '')
        try:
            time_parts = game_time.split(' - ')
            period = int(time_parts[0]) if len(time_parts) > 1 else None
            if len(time_parts) > 1:
                time_str = time_parts[1]
                mins, secs = time_str.split(':')
                minute = int(mins)
                second = int(secs)
            else:
                minute, second = None, None
        except:
            period, minute, second = None, None, None

        unified_corner = {
            'corner_id': corner['corner_id'],
            'source': 'soccernet',

            # Match info
            'match_id': corner.get('game_path'),
            'competition': competition,
            'season': season,
            'match_date': None,
            'home_team': None,
            'away_team': None,

            # Timing
            'period': period,
            'minute': minute,
            'second': second,

            # Team/Player
            'attacking_team': corner.get('team'),
            'player_name': None,

            # Location (unknown without tracking)
            'corner_x': None,
            'corner_y': None,
            'target_x': None,
            'target_y': None,

            # Outcome (to be labeled)
            'outcome_category': None,
            'outcome_type': None,
            'goal_scored': False,
            'time_to_outcome': None,

            # Data availability
            'has_player_positions': False,
            'has_tracking': False,
            'has_video': corner.get('has_video', False),
            'player_positions_json': None,
            'tracking_file': None,
            'video_path': corner.get('clip_path') if corner.get('clip_path') else None
        }

        unified_corners.append(unified_corner)

    unified_df = pd.DataFrame(unified_corners)

    print(f"  Created unified dataset with {len(unified_df)} corners")
    print(f"\n  Source breakdown:")
    print(f"    StatsBomb: {len(statsbomb_df)}")
    print(f"    SkillCorner: {len(skillcorner_df)}")
    print(f"    SoccerNet: {len(soccernet_df)}")

    return unified_df

def main():
    """Create unified corner dataset"""

    base_dir = Path("/home/mseo/CornerTactics")
    data_dir = base_dir / "data"
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Unified Corner Dataset Integration")
    print("="*60)
    print()

    # Load all data sources
    statsbomb_df = load_statsbomb_corners(data_dir)
    skillcorner_df = load_skillcorner_corners(data_dir)
    soccernet_df = load_soccernet_metadata(data_dir)

    # Create unified dataset
    unified_df = create_unified_schema(statsbomb_df, skillcorner_df, soccernet_df)

    # Save to parquet for efficiency
    output_file = output_dir / "unified_corners_dataset.parquet"
    unified_df.to_parquet(output_file, index=False)

    # Also save CSV for easy inspection
    csv_file = output_dir / "unified_corners_dataset.csv"
    unified_df.to_csv(csv_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("Integration Complete")
    print("="*60)
    print(f"Total corners: {len(unified_df)}")
    print(f"\nOutput files:")
    print(f"  Parquet: {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  CSV: {csv_file} ({csv_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Data availability summary
    print(f"\nData Availability:")
    print(f"  With player positions: {unified_df['has_player_positions'].sum()}")
    print(f"  With tracking data: {unified_df['has_tracking'].sum()}")
    print(f"  With video: {unified_df['has_video'].sum()}")
    print(f"  With outcomes: {unified_df['outcome_category'].notna().sum()}")

    # Outcome distribution (for corners with outcomes)
    if unified_df['outcome_category'].notna().any():
        print(f"\nOutcome Distribution (StatsBomb):")
        print(unified_df[unified_df['outcome_category'].notna()]['outcome_category'].value_counts())

    # Competition distribution
    print(f"\nCompetition Distribution:")
    print(unified_df['competition'].value_counts().head(10))

if __name__ == "__main__":
    main()
