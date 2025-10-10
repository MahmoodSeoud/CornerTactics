#!/usr/bin/env python3
"""
StatsBomb Data Loader
Extract corner kick events and outcomes from StatsBomb data.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsBombCornerLoader:
    """Load and process corner kick data from StatsBomb."""

    def __init__(self, output_dir: str = "data/statsbomb"):
        """Initialize loader with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import statsbombpy
        try:
            from statsbombpy import sb
            self.sb = sb
        except ImportError:
            raise ImportError("statsbombpy not installed. Run: pip install statsbombpy")

    def get_available_competitions(self) -> pd.DataFrame:
        """Get all available competitions from StatsBomb open data."""
        logger.info("Fetching available competitions...")
        competitions = self.sb.competitions()
        logger.info(f"Found {len(competitions)} competitions")
        return competitions

    def fetch_corner_events(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        Fetch all corner kick events for a competition/season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            DataFrame with corner kick events including coordinates
        """
        logger.info(f"Fetching events for competition {competition_id}, season {season_id}...")

        # Get all events for the competition
        events = self.sb.competition_events(
            country=None,  # Will use IDs directly
            division=None,
            season=None,
            gender=None,
            split=False,
            fmt='df'
        )

        # Filter for corner kicks
        # Corner kicks are Pass events with pass_type containing "Corner"
        corner_events = events[
            (events['type'] == 'Pass') &
            (events['pass_type'].notna()) &
            (events['pass_type'].str.contains('Corner', case=False, na=False))
        ].copy()

        logger.info(f"Found {len(corner_events)} corner kick events")

        return corner_events

    def fetch_competition_events(self, country: str, division: str,
                                 season: str, gender: str = "male") -> pd.DataFrame:
        """
        Fetch all events for a competition using country/division/season.

        Args:
            country: Country name (e.g., "England")
            division: Division name (e.g., "Premier League")
            season: Season name (e.g., "2019/2020")
            gender: Gender ("male" or "female")

        Returns:
            DataFrame with all events
        """
        logger.info(f"Fetching events for {country} {division} {season}...")

        events = self.sb.competition_events(
            country=country,
            division=division,
            season=season,
            gender=gender,
            split=False
        )

        logger.info(f"Fetched {len(events)} total events")
        logger.info(f"Events type: {type(events)}")
        return events

    def get_next_action(self, events_df: pd.DataFrame, corner_idx: int,
                       max_time_diff: float = 15.0) -> Optional[Dict]:
        """
        Find the next action after a corner kick.

        Args:
            events_df: Full events DataFrame
            corner_idx: Index of corner kick event
            max_time_diff: Maximum seconds to look ahead for outcome

        Returns:
            Dictionary with next action details or None
        """
        if corner_idx >= len(events_df) - 1:
            return None

        corner_event = events_df.iloc[corner_idx]
        corner_possession = corner_event.get('possession', None)
        corner_team = corner_event.get('team', None)
        corner_timestamp = corner_event.get('timestamp', None)

        # Look at subsequent events
        for i in range(corner_idx + 1, min(corner_idx + 20, len(events_df))):
            next_event = events_df.iloc[i]

            # Check time difference
            if corner_timestamp and next_event.get('timestamp'):
                # Parse timestamps if they're strings
                # Assuming format like "00:01:23.456"
                try:
                    corner_time = self._parse_timestamp(corner_timestamp)
                    next_time = self._parse_timestamp(next_event.get('timestamp'))
                    time_diff = next_time - corner_time

                    if time_diff > max_time_diff:
                        break
                except:
                    pass

            event_type = next_event.get('type', 'Unknown')

            # Key outcome events
            if event_type in ['Shot', 'Goal', 'Clearance', 'Interception',
                             'Duel', 'Foul Committed', 'Foul Won']:
                return {
                    'outcome_type': event_type,
                    'outcome_team': next_event.get('team', None),
                    'outcome_player': next_event.get('player', None),
                    'same_team': next_event.get('team') == corner_team,
                    'shot_outcome': next_event.get('shot_outcome', None) if event_type == 'Shot' else None,
                    'location': next_event.get('location', None),
                    'index_diff': i - corner_idx
                }

        # No clear outcome found
        return {
            'outcome_type': 'No Clear Outcome',
            'outcome_team': None,
            'outcome_player': None,
            'same_team': None,
            'shot_outcome': None,
            'location': None,
            'index_diff': None
        }

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to seconds."""
        if not isinstance(timestamp_str, str):
            return 0.0

        # Format: "HH:MM:SS.mmm"
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0

    def build_corner_dataset(self, country: str, division: str,
                            season: str, gender: str = "male") -> pd.DataFrame:
        """
        Build complete corner kick dataset with outcomes.

        Args:
            country: Country name
            division: Division name
            season: Season name
            gender: Gender

        Returns:
            DataFrame with corner events and outcomes
        """
        # Fetch all events
        events = self.fetch_competition_events(country, division, season, gender)

        # Filter corner kicks - pass_type can be string or NaN
        def is_corner(pass_type):
            if pd.isna(pass_type):
                return False
            if isinstance(pass_type, str):
                return 'corner' in pass_type.lower()
            return False

        corner_events = events[
            (events['type'] == 'Pass') &
            events['pass_type'].apply(is_corner)
        ].copy()

        logger.info(f"Processing {len(corner_events)} corner kicks...")

        # Extract features
        corners_data = []

        for idx, corner in corner_events.iterrows():
            # Find index in full events DataFrame
            corner_idx = events[events['id'] == corner['id']].index[0]

            # Get next action
            next_action = self.get_next_action(events, corner_idx)

            # Extract location coordinates
            location = corner.get('location', None)
            corner_x = location[0] if location and len(location) >= 2 else None
            corner_y = location[1] if location and len(location) >= 2 else None

            # Extract end location
            end_location = corner.get('pass_end_location', None)
            end_x = end_location[0] if end_location and len(end_location) >= 2 else None
            end_y = end_location[1] if end_location and len(end_location) >= 2 else None

            corner_data = {
                'match_id': corner.get('match_id', None),
                'period': corner.get('period', None),
                'minute': corner.get('minute', None),
                'second': corner.get('second', None),
                'team': corner.get('team', None),
                'player': corner.get('player', None),
                'corner_x': corner_x,
                'corner_y': corner_y,
                'end_x': end_x,
                'end_y': end_y,
                'pass_height': corner.get('pass_height', None),
                'pass_body_part': corner.get('pass_body_part', None),
                'pass_outcome': corner.get('pass_outcome', 'Complete'),
            }

            # Add next action data
            if next_action:
                corner_data.update({
                    'outcome_type': next_action.get('outcome_type'),
                    'outcome_team': next_action.get('outcome_team'),
                    'outcome_player': next_action.get('outcome_player'),
                    'same_team_outcome': next_action.get('same_team'),
                    'shot_outcome': next_action.get('shot_outcome'),
                    'events_to_outcome': next_action.get('index_diff')
                })

            corners_data.append(corner_data)

        df = pd.DataFrame(corners_data)
        logger.info(f"Built dataset with {len(df)} corner kicks")

        # Print outcome distribution
        if 'outcome_type' in df.columns:
            logger.info("\nOutcome distribution:")
            logger.info(df['outcome_type'].value_counts().to_string())

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset to CSV."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
