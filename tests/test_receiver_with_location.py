#!/usr/bin/env python3
"""
RED TEST: ReceiverLabeler should return event location along with player info.

This is needed to match receivers to freeze frame positions for both
attacking and defending players.
"""

import pytest
import pandas as pd
from src.receiver_labeler import ReceiverLabeler


def test_find_receiver_returns_location_for_ball_receipt():
    """
    RED TEST: find_receiver should return (player_id, player_name, location)
    for Ball Receipt events.
    """
    events = pd.DataFrame([
        {
            'id': '1',
            'index': 100,
            'type': 'Pass',
            'pass_type': 'Corner',
            'team': 'Team A',
            'player': 'Corner Taker',
            'player_id': 1000,
            'timestamp': '00:10:00.000',
            'location': [120.0, 40.0]  # Corner location
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Ball Receipt*',
            'team': 'Team A',
            'player': 'Striker',
            'player_id': 2000,
            'timestamp': '00:10:02.000',
            'location': [110.0, 38.0]  # Where striker receives ball
        }
    ])

    labeler = ReceiverLabeler()
    result = labeler.find_receiver(events, corner_event_id='1')

    # Should return tuple of (player_id, player_name, location)
    assert len(result) == 3, "Should return (player_id, player_name, location)"
    player_id, player_name, location = result

    assert player_id == 2000
    assert player_name == 'Striker'
    assert location is not None, "Should return event location"
    assert location == [110.0, 38.0], "Should return exact event location"


def test_find_receiver_returns_location_for_clearance():
    """
    RED TEST: find_receiver should return location for clearance events.
    """
    events = pd.DataFrame([
        {
            'id': '1',
            'index': 100,
            'type': 'Pass',
            'pass_type': 'Corner',
            'team': 'Team A',
            'player': 'Corner Taker',
            'player_id': 1000,
            'timestamp': '00:10:00.000',
            'location': [120.0, 40.0]
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Clearance',  # Defensive action
            'team': 'Team B',  # Defending team
            'player': 'Center Back',
            'player_id': 3000,
            'timestamp': '00:10:01.500',
            'location': [108.0, 42.0]  # Where defender clears from
        }
    ])

    labeler = ReceiverLabeler()
    result = labeler.find_receiver(events, corner_event_id='1')

    player_id, player_name, location = result

    assert player_id == 3000
    assert player_name == 'Center Back'
    assert location == [108.0, 42.0], "Should return clearance location"


def test_find_receiver_returns_none_location_when_missing():
    """
    RED TEST: find_receiver should return None for location if event has no location.
    """
    events = pd.DataFrame([
        {
            'id': '1',
            'index': 100,
            'type': 'Pass',
            'pass_type': 'Corner',
            'team': 'Team A',
            'player': 'Corner Taker',
            'player_id': 1000,
            'timestamp': '00:10:00.000'
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Duel',
            'team': 'Team B',
            'player': 'Defender',
            'player_id': 5000,
            'timestamp': '00:10:01.000'
            # No location field
        }
    ])

    labeler = ReceiverLabeler()
    result = labeler.find_receiver(events, corner_event_id='1')

    player_id, player_name, location = result

    assert player_id == 5000
    assert player_name == 'Defender'
    assert location is None, "Should return None for missing location"


def test_find_receiver_returns_none_when_no_receiver():
    """
    RED TEST: find_receiver should return (None, None, None) when no receiver found.
    """
    events = pd.DataFrame([
        {
            'id': '1',
            'index': 100,
            'type': 'Pass',
            'pass_type': 'Corner',
            'team': 'Team A',
            'player': 'Corner Taker',
            'player_id': 1000,
            'timestamp': '00:10:00.000'
        }
        # No subsequent events
    ])

    labeler = ReceiverLabeler()
    result = labeler.find_receiver(events, corner_event_id='1')

    assert result == (None, None, None), "Should return (None, None, None) when no receiver"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
