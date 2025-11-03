#!/usr/bin/env python3
"""
Tests for receiver labeling with defensive clearances.

Ensures that ReceiverLabeler can identify BOTH attacking and defending
players as receivers, matching TacticAI's methodology.
"""

import pytest
import pandas as pd
from src.receiver_labeler import ReceiverLabeler


def test_find_receiver_defensive_clearance():
    """
    RED TEST: Test that defensive clearances are labeled as receivers.

    This matches TacticAI's approach where the receiver can be ANY player
    (attacking OR defending) who makes first contact with the ball.
    """
    events = pd.DataFrame([
        {
            'id': '1',
            'index': 100,
            'type': 'Pass',
            'pass_type': 'Corner',
            'team': 'Team A',  # Attacking team
            'player': 'Corner Taker',
            'player_id': 1000,
            'timestamp': '00:10:00.000',
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Clearance',  # Defensive action
            'team': 'Team B',  # Defending team - DIFFERENT TEAM
            'player': 'Center Back',
            'player_id': 3000,  # Defender player_id
            'timestamp': '00:10:01.500',
            'minute': 10,
            'second': 1
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name = labeler.find_receiver(events, corner_event_id='1')

    # This should PASS if we correctly label defensive clearances
    assert receiver_id == 3000, "Should identify defending player who clears as receiver"
    assert receiver_name == 'Center Back', "Should return defender name"


def test_find_receiver_interception():
    """
    RED TEST: Test that defensive interceptions are labeled as receivers.
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
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Interception',  # Defensive action
            'team': 'Team B',  # Defending team
            'player': 'Goalkeeper',
            'player_id': 4000,
            'timestamp': '00:10:02.000',
            'minute': 10,
            'second': 2
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name = labeler.find_receiver(events, corner_event_id='1')

    assert receiver_id == 4000, "Should identify goalkeeper interception as receiver"
    assert receiver_name == 'Goalkeeper', "Should return goalkeeper name"


def test_find_receiver_duel():
    """
    RED TEST: Test that aerial duels (can be either team) are labeled as receivers.
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
            'minute': 10,
            'second': 0
        },
        {
            'id': '2',
            'index': 101,
            'type': 'Duel',  # Could be attacking or defending player
            'team': 'Team B',  # Defending team wins duel
            'player': 'Defender',
            'player_id': 5000,
            'timestamp': '00:10:01.000',
            'minute': 10,
            'second': 1
        }
    ])

    labeler = ReceiverLabeler()
    receiver_id, receiver_name = labeler.find_receiver(events, corner_event_id='1')

    assert receiver_id == 5000, "Should identify player who wins duel as receiver"
    assert receiver_name == 'Defender', "Should return player name from duel"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
