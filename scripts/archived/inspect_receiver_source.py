#!/usr/bin/env python3
"""
Check where receiver information comes from in StatsBomb data.
"""

import json
import requests

# Champions League Final 2021
match_id = 3795506

# Get events
events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
print(f"Fetching events for match {match_id}...")
response = requests.get(events_url)
events = response.json()

# Find first corner kick
corner_event = None
for event in events:
    if event.get('type', {}).get('name') == 'Pass' and event.get('pass', {}).get('type', {}).get('name') == 'Corner':
        corner_event = event
        break

if corner_event:
    print(f"\nFound corner kick:")
    print(f"Event ID: {corner_event['id']}")
    print(f"Player: {corner_event.get('player', {}).get('name')}")
    print(f"Team: {corner_event.get('team', {}).get('name')}")

    # Check if pass has recipient
    if 'pass' in corner_event:
        pass_data = corner_event['pass']
        print(f"\nPass data keys: {list(pass_data.keys())}")

        if 'recipient' in pass_data:
            print(f"\n✓ Recipient found:")
            print(f"  Name: {pass_data['recipient'].get('name')}")
            print(f"  ID: {pass_data['recipient'].get('id')}")
        else:
            print(f"\n✗ No recipient field in pass data")

    # Show next few events to see what happens after corner
    corner_idx = events.index(corner_event)
    print(f"\n\nNext 3 events after corner:")
    for i in range(1, 4):
        if corner_idx + i < len(events):
            next_event = events[corner_idx + i]
            print(f"\n{i}. {next_event.get('type', {}).get('name')}")
            print(f"   Player: {next_event.get('player', {}).get('name')}")
            print(f"   Team: {next_event.get('team', {}).get('name')}")

            # Check if event has location
            if 'location' in next_event:
                print(f"   Location: {next_event['location']}")

            # If it's a pass, check recipient
            if 'pass' in next_event and 'recipient' in next_event['pass']:
                print(f"   Recipient: {next_event['pass']['recipient'].get('name')}")

    # Check if the Ball Receipt event has location
    print(f"\n\nBall Receipt event (full data):")
    if corner_idx + 1 < len(events):
        ball_receipt = events[corner_idx + 1]
        print(json.dumps(ball_receipt, indent=2))
