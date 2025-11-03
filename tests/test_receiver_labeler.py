#!/usr/bin/env python3
"""
Tests for receiver label extraction.

Tests the ReceiverLabeler class that identifies which player receives
the ball after a corner kick.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_find_receiver_from_events():
    """Test identifying receiver from a sequence of StatsBomb events."""
    from src.receiver_labeler import ReceiverLabeler

    # Create mock events DataFrame with corner and subsequent events
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
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Pass',
            'pass_type': 'Head Pass',
            'team': 'Team A',
            'player': 'Striker',
            'player_id': 2000,
            'timestamp': '00:10:02.500',
            'minute': 10,
            'second': 2
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name, location = labeler.find_receiver(events, corner_event_id='1')

    assert receiver_id == 2000, "Should identify Striker as receiver"
    assert receiver_name == 'Striker', "Should return receiver name"


def test_find_receiver_within_time_window():
    """Test that receiver must be within 0-5 second window."""
    from src.receiver_labeler import ReceiverLabeler

    # Event outside 5 second window
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
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Shot',
            'team': 'Team A',
            'player': 'Striker',
            'player_id': 2000,
            'timestamp': '00:10:06.000',  # 6 seconds later - outside window
            'minute': 10,
            'second': 6
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name, location = labeler.find_receiver(events, corner_event_id='1', max_time_diff=5.0)

    assert receiver_id is None, "Should return None for events outside 5 second window"


def test_receiver_excludes_corner_taker():
    """Test that corner taker is excluded as receiver."""
    from src.receiver_labeler import ReceiverLabeler

    # Corner taker touches ball again (short corner)
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
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Pass',
            'team': 'Team A',
            'player': 'Corner Taker',  # Same player
            'player_id': 1000,
            'timestamp': '00:10:01.000',
            'minute': 10,
            'second': 1
        },
        {
            'id': '3',
            'index': 102,
            'type': 'Shot',
            'team': 'Team A',
            'player': 'Striker',
            'player_id': 2000,
            'timestamp': '00:10:03.000',
            'minute': 10,
            'second': 3
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name, location = labeler.find_receiver(events, corner_event_id='1')

    # Should skip corner taker and find Striker as receiver
    assert receiver_id == 2000, "Should skip corner taker and identify next player"


def test_receiver_valid_event_types():
    """Test that only certain event types count as 'receiving' the ball."""
    from src.receiver_labeler import ReceiverLabeler

    # Test various event types
    valid_events = ['Pass', 'Shot', 'Duel', 'Interception', 'Clearance', 'Miscontrol']

    for event_type in valid_events:
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
                'minute': 10,
                'second': 0
            },
            {
                'id': '2',
                'index': 101,
                'type': event_type,
                'team': 'Team A',
                'player': 'Receiver',
                'player_id': 2000,
                'timestamp': '00:10:02.000',
                'minute': 10,
                'second': 2
            }
        ])

        labeler = ReceiverLabeler()
        receiver_id, receiver_name, location = labeler.find_receiver(events, corner_event_id='1')

        assert receiver_id == 2000, f"Event type '{event_type}' should be valid for receiver"


def test_receiver_no_subsequent_events():
    """Test handling when there are no events after corner."""
    from src.receiver_labeler import ReceiverLabeler

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
            'minute': 10,
            'second': 0
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name, location = labeler.find_receiver(events, corner_event_id='1')

    assert receiver_id is None, "Should return None when no subsequent events"
