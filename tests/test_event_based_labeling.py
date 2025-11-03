#!/usr/bin/env python3
"""
RED TESTS: Event-stream-based receiver labeling.

Tests the new approach that uses ReceiverLabeler with event streams
to label ALL corners (including defensive clearances).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock


def test_extract_receiver_from_events_attacking():
    """
    RED TEST: Extract receiver from events for attacking Ball Receipt.

    Should use ReceiverLabeler to find receiver and return location
    for position matching.
    """
    from src.receiver_labeler import ReceiverLabeler

    # Mock corner event and subsequent Ball Receipt
    events_df = pd.DataFrame([
        {
            'id': 'corner_123',
            'type': 'Pass',
            'pass_type': 'Corner',
            'player': 'Corner Taker',
            'player_id': 1000,
            'team': 'Team A',
            'timestamp': '00:10:00.000',
            'location': [120.0, 40.0]
        },
        {
            'id': 'receipt_456',
            'type': 'Ball Receipt*',
            'player': 'Striker',
            'player_id': 2000,
            'team': 'Team A',
            'timestamp': '00:10:02.000',
            'location': [110.0, 38.5]  # Where striker receives
        }
    ])

    labeler = ReceiverLabeler()
    player_id, player_name, location = labeler.find_receiver(
        events_df,
        corner_event_id='corner_123'
    )

    # Should find attacking receiver
    assert player_id == 2000
    assert player_name == 'Striker'
    assert location == [110.0, 38.5]


def test_extract_receiver_from_events_defensive_clearance():
    """
    RED TEST: Extract receiver from events for defensive clearance.

    This is the KEY test - ensuring we can label defensive players
    as receivers to increase coverage.
    """
    from src.receiver_labeler import ReceiverLabeler

    # Mock corner event and subsequent defensive clearance
    events_df = pd.DataFrame([
        {
            'id': 'corner_123',
            'type': 'Pass',
            'pass_type': 'Corner',
            'player': 'Corner Taker',
            'player_id': 1000,
            'team': 'Team A',  # Attacking
            'timestamp': '00:10:00.000',
            'location': [120.0, 40.0]
        },
        {
            'id': 'clearance_789',
            'type': 'Clearance',
            'player': 'Center Back',
            'player_id': 3000,
            'team': 'Team B',  # Defending - DIFFERENT TEAM
            'timestamp': '00:10:01.500',
            'location': [108.0, 42.0]  # Where defender clears from
        }
    ])

    labeler = ReceiverLabeler()
    player_id, player_name, location = labeler.find_receiver(
        events_df,
        corner_event_id='corner_123'
    )

    # Should find DEFENSIVE receiver (clearance)
    assert player_id == 3000
    assert player_name == 'Center Back'
    assert location == [108.0, 42.0]
    # This is the crucial test - defensive clearance is labeled!


def test_match_receiver_location_to_freeze_frame_attacking():
    """
    RED TEST: Match receiver location to closest freeze frame position.

    Should find the closest player position in the freeze frame data,
    prioritizing the attacking team when receiver is attacking.
    """
    # Mock freeze frame positions (as they appear in graphs)
    freeze_frame_positions = np.array([
        [110.2, 38.3],   # Index 0 - Close to receiver location
        [105.0, 45.0],   # Index 1
        [112.0, 35.0],   # Index 2
        [95.0, 40.0],    # Index 3
    ])

    # Mock team labels
    teams = ['attacking', 'attacking', 'defending', 'defending']

    # Receiver location from event
    receiver_location = np.array([110.0, 38.5])
    receiver_team = 'attacking'  # From event data

    # Find closest position (should prioritize attacking team)
    # This is the function we need to implement
    from scripts.preprocessing.add_receiver_labels_v2 import match_location_to_node

    node_index = match_location_to_node(
        receiver_location,
        freeze_frame_positions,
        teams,
        receiver_team
    )

    # Should match to index 0 (closest attacking player)
    assert node_index == 0


def test_match_receiver_location_to_freeze_frame_defending():
    """
    RED TEST: Match defensive clearance location to freeze frame.

    NEW CAPABILITY: Should match defending player who clears the ball.
    """
    freeze_frame_positions = np.array([
        [110.0, 38.0],   # Index 0 - Attacking
        [105.0, 45.0],   # Index 1 - Attacking
        [108.2, 41.8],   # Index 2 - Defending (close to clearance)
        [95.0, 40.0],    # Index 3 - Defending
    ])

    teams = ['attacking', 'attacking', 'defending', 'defending']

    # Receiver is defender who clears
    receiver_location = np.array([108.0, 42.0])
    receiver_team = 'defending'

    from scripts.preprocessing.add_receiver_labels_v2 import match_location_to_node

    node_index = match_location_to_node(
        receiver_location,
        freeze_frame_positions,
        teams,
        receiver_team
    )

    # Should match to index 2 (closest defending player)
    assert node_index == 2


def test_match_location_no_team_filter():
    """
    RED TEST: When team is unknown, match to ANY closest player.
    """
    freeze_frame_positions = np.array([
        [110.0, 38.0],   # Index 0
        [108.1, 42.1],   # Index 1 - Closest overall
        [95.0, 40.0],    # Index 2
    ])

    teams = ['attacking', 'defending', 'defending']

    receiver_location = np.array([108.0, 42.0])
    receiver_team = None  # Unknown team

    from scripts.preprocessing.add_receiver_labels_v2 import match_location_to_node

    node_index = match_location_to_node(
        receiver_location,
        freeze_frame_positions,
        teams,
        receiver_team
    )

    # Should match to index 1 (closest overall, any team)
    assert node_index == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
