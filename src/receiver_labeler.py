#!/usr/bin/env python3
"""
Receiver Label Extraction for TacticAI Corner Kick Prediction

Identifies which player receives the ball after a corner kick by analyzing
subsequent StatsBomb events within a 0-5 second window.

Based on TacticAI Implementation Plan Day 1-2: Receiver Label Extraction
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReceiverLabeler:
    """
    Identifies the receiver of a corner kick from StatsBomb event data.

    The receiver is defined as the first player (excluding the corner taker)
    who touches the ball within 0-5 seconds after the corner kick event.
    """

    # Event types that count as "receiving" the ball
    VALID_RECEIVER_EVENTS = [
        'Pass', 'Shot', 'Duel', 'Interception',
        'Clearance', 'Miscontrol', 'Ball Receipt*'
    ]

    def __init__(self):
        """Initialize receiver labeler."""
        pass

    def find_receiver(
        self,
        events_df: pd.DataFrame,
        corner_event_id: str,
        max_time_diff: float = 5.0
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the receiver of a corner kick.

        Args:
            events_df: DataFrame with StatsBomb events (must include corner event)
            corner_event_id: ID of the corner kick event
            max_time_diff: Maximum seconds after corner to look for receiver (default: 5.0)

        Returns:
            Tuple of (receiver_player_id, receiver_player_name) or (None, None) if not found
        """
        # Find the corner event
        corner_mask = events_df['id'] == corner_event_id
        if not corner_mask.any():
            logger.warning(f"Corner event {corner_event_id} not found in events")
            return None, None

        corner_idx = events_df[corner_mask].index[0]
        corner_event = events_df.loc[corner_idx]

        corner_taker_id = corner_event.get('player_id')
        corner_timestamp = corner_event.get('timestamp')
        corner_time = self._parse_timestamp(corner_timestamp)

        # Look at subsequent events
        subsequent_events = events_df.loc[corner_idx + 1:]

        for idx, event in subsequent_events.iterrows():
            # Check time window
            event_timestamp = event.get('timestamp')
            if event_timestamp and corner_time is not None:
                event_time = self._parse_timestamp(event_timestamp)
                if event_time is not None:
                    time_diff = event_time - corner_time
                    if time_diff > max_time_diff:
                        # Exceeded time window
                        break

            # Check if this is a valid receiver event type
            event_type = event.get('type')
            if not self._is_valid_receiver_event(event_type):
                continue

            # Get player who performed this action
            player_id = event.get('player_id')
            player_name = event.get('player')

            # Exclude corner taker (short corners)
            if player_id == corner_taker_id:
                continue

            # Found the receiver!
            if player_id is not None:
                return int(player_id), player_name

        # No receiver found within time window
        return None, None

    def _is_valid_receiver_event(self, event_type: str) -> bool:
        """Check if event type counts as receiving the ball."""
        if event_type is None:
            return False

        # Check for exact matches
        if event_type in self.VALID_RECEIVER_EVENTS:
            return True

        # Check for partial matches (e.g., "Ball Receipt*" matches "Ball Receipt")
        for valid_event in self.VALID_RECEIVER_EVENTS:
            if valid_event.endswith('*') and event_type.startswith(valid_event[:-1]):
                return True

        return False

    def _parse_timestamp(self, timestamp: str) -> Optional[float]:
        """
        Parse StatsBomb timestamp to seconds.

        Args:
            timestamp: Timestamp string in format "HH:MM:SS.mmm"

        Returns:
            Total seconds as float, or None if parsing fails
        """
        if timestamp is None or pd.isna(timestamp):
            return None

        try:
            # Format: "00:10:23.456"
            parts = timestamp.split(':')
            if len(parts) != 3:
                return None

            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse timestamp: {timestamp}")
            return None
