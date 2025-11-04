#!/usr/bin/env python3
"""
Unified Outcome Labeling Module for Corner Kick Analysis

This module provides outcome classification for corner kicks across multiple data sources:
- StatsBomb: Event data with 360 freeze frames
- SkillCorner: Continuous tracking with dynamic events
- SoccerNet: Video clips with label data

Implements Phase 1.2 of the Corner GNN project plan.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CornerOutcome:
    """
    Standardized corner kick outcome data structure

    Attributes:
        outcome_category: High-level category (Goal/Shot/Clearance/Possession/Loss)
        outcome_type: Detailed type (e.g., "Shot - Saved", "Clearance", etc.)
        outcome_team: Team that performed the outcome action
        outcome_player: Player who performed the action (if available)
        same_team: Whether outcome was by attacking team (True) or defending (False)
        time_to_outcome: Time in seconds from corner to outcome
        events_to_outcome: Number of events between corner and outcome
        goal_scored: Boolean flag for goals
        shot_outcome: Specific shot result (Goal/Saved/Blocked/Off T/Wayward/Post)
        outcome_location: (x, y) coordinates of outcome event
        xthreat_delta: Change in expected threat from corner to outcome
    """
    outcome_category: str
    outcome_type: str
    outcome_team: Optional[str] = None
    outcome_player: Optional[str] = None
    same_team: Optional[bool] = None
    time_to_outcome: Optional[float] = None
    events_to_outcome: Optional[int] = None
    goal_scored: bool = False
    shot_outcome: Optional[str] = None
    outcome_location: Optional[Tuple[float, float]] = None
    xthreat_delta: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame insertion"""
        return {
            'outcome_category': self.outcome_category,
            'outcome_type': self.outcome_type,
            'outcome_team': self.outcome_team,
            'outcome_player': self.outcome_player,
            'same_team': self.same_team,
            'time_to_outcome': self.time_to_outcome,
            'events_to_outcome': self.events_to_outcome,
            'goal_scored': self.goal_scored,
            'shot_outcome': self.shot_outcome,
            'outcome_location': self.outcome_location,
            'xthreat_delta': self.xthreat_delta
        }


class OutcomeLabeler:
    """Base class for corner kick outcome labeling"""

    def __init__(self, max_time_window: float = 20.0):
        """
        Initialize outcome labeler

        Args:
            max_time_window: Maximum time in seconds to search for outcomes
        """
        self.max_time_window = max_time_window

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> float:
        """
        Convert timestamp string to seconds

        Supports formats:
        - HH:MM:SS.mmm (StatsBomb)
        - Seconds as float (SkillCorner)
        - Various other formats

        Args:
            timestamp_str: Timestamp string

        Returns:
            Time in seconds as float
        """
        if pd.isna(timestamp_str) or timestamp_str == '':
            return 0.0

        # Already a number
        if isinstance(timestamp_str, (int, float)):
            return float(timestamp_str)

        # Format: "HH:MM:SS.mmm"
        parts = str(timestamp_str).split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        # Try direct float conversion
        try:
            return float(timestamp_str)
        except:
            return 0.0

    @staticmethod
    def calculate_xthreat_delta(
        corner_location: Optional[Tuple[float, float]],
        outcome_location: Optional[Tuple[float, float]],
        pitch_length: float = 120.0,
        pitch_width: float = 80.0
    ) -> float:
        """
        Calculate change in expected threat (xThreat)

        Simplified xThreat based on distance to goal. Positive values mean
        increased threat (moved closer to goal).

        Args:
            corner_location: (x, y) of corner kick
            outcome_location: (x, y) of outcome event
            pitch_length: Pitch length (default: 120 for StatsBomb)
            pitch_width: Pitch width (default: 80 for StatsBomb)

        Returns:
            xThreat delta (positive = more shot)
        """
        if not corner_location or not outcome_location:
            return 0.0

        try:
            corner_x, corner_y = corner_location
            outcome_x, outcome_y = outcome_location

            # Goal at (pitch_length, pitch_width/2)
            goal_x, goal_y = pitch_length, pitch_width / 2

            # Distance to goal
            corner_dist = np.sqrt((goal_x - corner_x)**2 + (goal_y - corner_y)**2)
            outcome_dist = np.sqrt((goal_x - outcome_x)**2 + (goal_y - outcome_y)**2)

            # Positive value = moved closer to goal (more shot)
            return corner_dist - outcome_dist
        except:
            return 0.0

    @staticmethod
    def calculate_angle_to_goal(
        location: Tuple[float, float],
        pitch_length: float = 120.0,
        pitch_width: float = 80.0
    ) -> float:
        """
        Calculate angle to goal from a location

        Args:
            location: (x, y) position
            pitch_length: Pitch length
            pitch_width: Pitch width

        Returns:
            Angle in degrees
        """
        x, y = location
        goal_x, goal_y = pitch_length, pitch_width / 2

        dx = goal_x - x
        dy = goal_y - y

        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle


class StatsBombOutcomeLabeler(OutcomeLabeler):
    """Outcome labeler for StatsBomb event data"""

    def label_corner_outcome(
        self,
        events_df: pd.DataFrame,
        corner_idx: int
    ) -> CornerOutcome:
        """
        Label outcome for a single corner kick

        Args:
            events_df: Full events DataFrame for the match
            corner_idx: Index of corner event in events_df

        Returns:
            CornerOutcome object with classification
        """
        corner_event = events_df.iloc[corner_idx]
        corner_team = corner_event.get('team')
        corner_time_sec = self.parse_timestamp(corner_event.get('timestamp'))
        corner_location = corner_event.get('pass_end_location')

        # Default outcome (if nothing found within time window)
        default_outcome = CornerOutcome(
            outcome_category='Possession',
            outcome_type='Maintained Possession',
            same_team=True
        )

        # Get ALL events after corner and filter by time window
        # This is more reliable than scanning sequentially (which may miss events)
        events_after = events_df.iloc[corner_idx + 1:].copy()

        # Calculate time differences for all events
        events_after['time_sec'] = events_after['timestamp'].apply(self.parse_timestamp)
        events_after['time_diff'] = events_after['time_sec'] - corner_time_sec

        # Filter to events within time window
        events_in_window = events_after[
            (events_after['time_diff'] > 0) &  # Only events AFTER corner
            (events_after['time_diff'] <= self.max_time_window)
        ].copy()

        # Sort by time to process in chronological order
        events_in_window = events_in_window.sort_values('time_diff')

        # Scan through events in time window
        for idx, next_event in events_in_window.iterrows():
            time_diff = next_event['time_diff']
            event_type = next_event.get('type')
            next_team = next_event.get('team')
            same_team = (next_team == corner_team)
            location = next_event.get('location')

            # Calculate events_to_outcome for all outcomes
            events_to_outcome = idx - corner_idx if isinstance(idx, int) else None

            # PRIORITY 1: Shot (Goal or attempt)
            if event_type == 'Shot':
                shot_outcome_val = next_event.get('shot_outcome')

                if shot_outcome_val == 'Goal':
                    return CornerOutcome(
                        outcome_category='Goal',
                        outcome_type='Shot - Goal',
                        outcome_team=next_team,
                        outcome_player=next_event.get('player'),
                        same_team=same_team,
                        time_to_outcome=time_diff,
                        events_to_outcome=events_to_outcome,
                        goal_scored=True,
                        shot_outcome=shot_outcome_val,
                        outcome_location=location,
                        xthreat_delta=self.calculate_xthreat_delta(corner_location, location)
                    )
                else:
                    # Shot but no goal
                    return CornerOutcome(
                        outcome_category='Shot',
                        outcome_type=f'Shot - {shot_outcome_val}',
                        outcome_team=next_team,
                        outcome_player=next_event.get('player'),
                        same_team=same_team,
                        time_to_outcome=time_diff,
                        events_to_outcome=events_to_outcome,
                        goal_scored=False,
                        shot_outcome=shot_outcome_val,
                        outcome_location=location,
                        xthreat_delta=self.calculate_xthreat_delta(corner_location, location)
                    )

            # PRIORITY 2: Defensive clearance (by defending team)
            if event_type == 'Clearance' and not same_team:
                return CornerOutcome(
                    outcome_category='Clearance',
                    outcome_type='Clearance',
                    outcome_team=next_team,
                    outcome_player=next_event.get('player'),
                    same_team=False,
                    time_to_outcome=time_diff,
                    events_to_outcome=events_to_outcome,
                    outcome_location=location,
                    xthreat_delta=self.calculate_xthreat_delta(corner_location, location)
                )

            # PRIORITY 3: Interception (possession lost)
            if event_type == 'Interception' and not same_team:
                return CornerOutcome(
                    outcome_category='Loss',
                    outcome_type='Interception',
                    outcome_team=next_team,
                    outcome_player=next_event.get('player'),
                    same_team=False,
                    time_to_outcome=time_diff,
                    events_to_outcome=events_to_outcome,
                    outcome_location=location,
                    xthreat_delta=self.calculate_xthreat_delta(corner_location, location)
                )

            # PRIORITY 4: Second corner (retained possession)
            if same_team and event_type == 'Pass':
                pass_type = next_event.get('pass_type', '')
                if isinstance(pass_type, str) and 'corner' in pass_type.lower():
                    return CornerOutcome(
                        outcome_category='Possession',
                        outcome_type='Second Corner',
                        outcome_team=next_team,
                        outcome_player=next_event.get('player'),
                        same_team=True,
                        time_to_outcome=time_diff,
                        events_to_outcome=events_to_outcome
                    )

            # PRIORITY 5: Other possession loss
            if event_type in ['Foul Won', 'Duel'] and not same_team:
                return CornerOutcome(
                    outcome_category='Loss',
                    outcome_type=event_type,
                    outcome_team=next_team,
                    outcome_player=next_event.get('player'),
                    same_team=False,
                    time_to_outcome=time_diff,
                    events_to_outcome=events_to_outcome,
                    outcome_location=location
                )

        # No clear outcome found - possession maintained
        return default_outcome


class SkillCornerOutcomeLabeler(OutcomeLabeler):
    """Outcome labeler for SkillCorner tracking data"""

    def label_corner_outcome(
        self,
        dynamic_events_df: pd.DataFrame,
        corner_event: pd.Series,
        phases_df: Optional[pd.DataFrame] = None
    ) -> CornerOutcome:
        """
        Label outcome for a SkillCorner corner kick

        Args:
            dynamic_events_df: Dynamic events DataFrame for the match
            corner_event: Series containing the corner event
            phases_df: Optional phases of play DataFrame

        Returns:
            CornerOutcome object with classification
        """
        corner_frame = corner_event['frame_start']
        corner_time = corner_event['time_start']
        corner_team = corner_event.get('team_shortname')

        # Calculate time threshold in frames (10 fps)
        max_frames = int(self.max_time_window * 10)

        # SkillCorner uses player_possession events with end_type field
        # Find player_possession events following the corner
        following_possessions = dynamic_events_df[
            (dynamic_events_df['frame_start'] > corner_frame) &
            (dynamic_events_df['frame_start'] <= corner_frame + max_frames) &
            (dynamic_events_df['event_type'] == 'player_possession')
        ].sort_values('frame_start')

        # Priority: Shot > Clearance > Loss > Possession
        for _, possession in following_possessions.iterrows():
            end_type = possession.get('end_type', '')
            if pd.isna(end_type):
                continue

            event_team = possession.get('team_shortname')
            same_team = (event_team == corner_team)

            frame_diff = possession['frame_start'] - corner_frame
            time_diff = frame_diff / 10.0  # 10 fps

            # Check for shots (highest priority)
            if end_type == 'shot':
                return CornerOutcome(
                    outcome_category='Shot',
                    outcome_type='Shot',
                    outcome_team=event_team,
                    same_team=same_team,
                    time_to_outcome=time_diff,
                    outcome_location=(possession.get('x_end'), possession.get('y_end'))
                )

            # Check for clearances (defending team clears)
            if end_type == 'clearance':
                return CornerOutcome(
                    outcome_category='Clearance',
                    outcome_type='Clearance',
                    outcome_team=event_team,
                    same_team=same_team,
                    time_to_outcome=time_diff,
                    outcome_location=(possession.get('x_end'), possession.get('y_end'))
                )

            # Check for possession loss
            if end_type == 'possession_loss' or end_type == 'indirect_disruption' or end_type == 'direct_disruption':
                # If defending team gains possession, it's a clearance/loss
                if not same_team:
                    return CornerOutcome(
                        outcome_category='Loss',
                        outcome_type='Possession Loss',
                        outcome_team=event_team,
                        same_team=False,
                        time_to_outcome=time_diff,
                        outcome_location=(possession.get('x_end'), possession.get('y_end'))
                    )

        # Check for early possession change to opposing team (indirect indicator of clearance)
        # Look at ALL events, not just possessions
        all_events = dynamic_events_df[
            (dynamic_events_df['frame_start'] > corner_frame) &
            (dynamic_events_df['frame_start'] <= corner_frame + max_frames)
        ].sort_values('frame_start')

        # Find first event by opposing team
        opposing_events = all_events[all_events['team_shortname'] != corner_team]
        if len(opposing_events) > 0:
            first_opp = opposing_events.iloc[0]
            frame_diff = first_opp['frame_start'] - corner_frame
            time_diff = frame_diff / 10.0

            # If opposing team gets involved early (< 5 seconds), it's likely a clearance
            if time_diff < 5.0:
                return CornerOutcome(
                    outcome_category='Clearance',
                    outcome_type='Early Opposition Action',
                    outcome_team=first_opp.get('team_shortname'),
                    same_team=False,
                    time_to_outcome=time_diff,
                    outcome_location=(first_opp.get('x_start'), first_opp.get('y_start'))
                )

        # Default: possession maintained
        return CornerOutcome(
            outcome_category='Possession',
            outcome_type='Maintained Possession',
            same_team=True
        )


class SoccerNetOutcomeLabeler(OutcomeLabeler):
    """Outcome labeler for SoccerNet video data"""

    def label_corner_outcome(
        self,
        labels_data: Dict,
        corner_timestamp: float,
        corner_team: str
    ) -> CornerOutcome:
        """
        Label outcome for a SoccerNet corner

        Args:
            labels_data: Parsed labels JSON for the match
            corner_timestamp: Timestamp of corner in seconds
            corner_team: Team taking the corner

        Returns:
            CornerOutcome object with classification
        """
        # SoccerNet labels structure: annotations with labels and timestamps
        annotations = labels_data.get('annotations', [])

        # Find events following the corner within time window
        for annotation in annotations:
            event_time = annotation.get('gameTime', '')
            label = annotation.get('label', '')
            team = annotation.get('team', '')

            # Parse SoccerNet time format (e.g., "1 - 12:34")
            try:
                parts = event_time.split(' - ')
                if len(parts) == 2:
                    half = int(parts[0])
                    time_parts = parts[1].split(':')
                    minutes = int(time_parts[0])
                    seconds = int(time_parts[1])
                    event_timestamp = (half - 1) * 2700 + minutes * 60 + seconds
                else:
                    continue
            except:
                continue

            time_diff = event_timestamp - corner_timestamp

            # Check if within time window and after corner
            if time_diff < 0 or time_diff > self.max_time_window:
                continue

            same_team = (team == corner_team)

            # Check for goals
            if 'goal' in label.lower():
                return CornerOutcome(
                    outcome_category='Goal',
                    outcome_type='Shot - Goal',
                    outcome_team=team,
                    same_team=same_team,
                    time_to_outcome=time_diff,
                    goal_scored=True
                )

            # Check for shots
            if 'shot' in label.lower():
                return CornerOutcome(
                    outcome_category='Shot',
                    outcome_type='Shot',
                    outcome_team=team,
                    same_team=same_team,
                    time_to_outcome=time_diff
                )

            # Check for clearances
            if 'clearance' in label.lower() and not same_team:
                return CornerOutcome(
                    outcome_category='Clearance',
                    outcome_type='Clearance',
                    outcome_team=team,
                    same_team=False,
                    time_to_outcome=time_diff
                )

        # Default: possession maintained
        return CornerOutcome(
            outcome_category='Possession',
            outcome_type='Maintained Possession',
            same_team=True
        )


def calculate_success_metrics(outcomes_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for corner outcomes

    Args:
        outcomes_df: DataFrame with outcome labels

    Returns:
        Dictionary of success metrics
    """
    total_corners = len(outcomes_df)

    if total_corners == 0:
        return {}

    goals = len(outcomes_df[outcomes_df['goal_scored'] == True])
    shots = len(outcomes_df[outcomes_df['outcome_category'].isin(['Goal', 'Shot'])])
    clearances = len(outcomes_df[outcomes_df['outcome_category'] == 'Clearance'])
    possession = len(outcomes_df[outcomes_df['outcome_category'] == 'Possession'])

    metrics = {
        'total_corners': total_corners,
        'goals': goals,
        'goal_rate': goals / total_corners,
        'shots': shots,
        'shot_rate': shots / total_corners,
        'clearances': clearances,
        'clearance_rate': clearances / total_corners,
        'possession_maintained': possession,
        'possession_rate': possession / total_corners
    }

    # Time to outcome stats
    outcome_times = outcomes_df[outcomes_df['time_to_outcome'].notna()]['time_to_outcome']
    if len(outcome_times) > 0:
        metrics['avg_time_to_outcome'] = outcome_times.mean()
        metrics['median_time_to_outcome'] = outcome_times.median()

    return metrics
