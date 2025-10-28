#!/usr/bin/env python3
"""
Test matching receiver to freeze frame position using Ball Receipt location.
"""

import json
import requests
import numpy as np

# Champions League Final 2021
match_id = 3795506

# Get events and 360 data
print(f"Fetching data for match {match_id}...")
events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
frames_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"

events = requests.get(events_url).json()
freeze_frames_all = requests.get(frames_url).json()

# Build set of event IDs that have freeze frames
freeze_frame_event_ids = {ff.get('event_uuid') for ff in freeze_frames_all}
print(f"✓ {len(freeze_frame_event_ids)} events have freeze frames")

# Find first corner with recipient AND freeze frame
corner_event = None
for event in events:
    if (event.get('type', {}).get('name') == 'Pass' and
        event.get('pass', {}).get('type', {}).get('name') == 'Corner' and
        'recipient' in event.get('pass', {}) and
        event['id'] in freeze_frame_event_ids):
        corner_event = event
        break

if not corner_event:
    print("No corner with recipient found")
    exit(1)

corner_id = corner_event['id']
recipient_name = corner_event['pass']['recipient']['name']
recipient_id = corner_event['pass']['recipient']['id']

print(f"\n✓ Found corner kick:")
print(f"  Event ID: {corner_id}")
print(f"  Kicker: {corner_event['player']['name']}")
print(f"  Recipient: {recipient_name} (ID: {recipient_id})")

# Find Ball Receipt event
ball_receipt = None
for event in events:
    if (event.get('type', {}).get('name') == 'Ball Receipt*' and
        event.get('player', {}).get('id') == recipient_id and
        corner_id in event.get('related_events', [])):
        ball_receipt = event
        break

if ball_receipt and 'location' in ball_receipt:
    receipt_location = np.array(ball_receipt['location'])
    print(f"\n✓ Ball Receipt location: {receipt_location}")

    # Find freeze frame for this corner
    freeze_frame_data = None
    for ff in freeze_frames_all:
        if ff.get('event_uuid') == corner_id:
            freeze_frame_data = ff['freeze_frame']
            break

    if freeze_frame_data:
        print(f"\n✓ Found freeze frame with {len(freeze_frame_data)} players")

        # Filter to attacking players (teammates)
        attacking_players = [p for p in freeze_frame_data if p['teammate']]
        print(f"  Attacking players: {len(attacking_players)}")

        # Find closest attacking player to receipt location
        min_distance = float('inf')
        closest_idx = None
        for idx, player in enumerate(attacking_players):
            player_loc = np.array(player['location'])
            distance = np.linalg.norm(player_loc - receipt_location)

            if distance < min_distance:
                min_distance = distance
                closest_idx = idx

        if closest_idx is not None:
            print(f"\n✓ Closest attacking player to receiver:")
            print(f"  Index in attacking team: {closest_idx}")
            print(f"  Position: {attacking_players[closest_idx]['location']}")
            print(f"  Distance from receipt location: {min_distance:.2f} meters")
            print(f"  Is keeper: {attacking_players[closest_idx]['keeper']}")

            print(f"\n✓ SUCCESS: Can map receiver '{recipient_name}' to node index {closest_idx}")
        else:
            print("\n✗ No attacking players in freeze frame")
    else:
        print(f"\n✗ No freeze frame found for corner {corner_id}")
else:
    print("\n✗ No Ball Receipt event with location found")
