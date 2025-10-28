#!/usr/bin/env python3
"""
Inspect raw StatsBomb 360 JSON to understand freeze frame structure.
"""

import json
import requests

# Find a match with 360 data from StatsBomb open data
# Champions League Final 2021: 3795506
match_id = 3795506

url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"

print(f"Fetching 360 data for match {match_id}...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"✓ Found {len(data)} freeze frames\n")

    # Find first corner kick in the events
    events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    events_response = requests.get(events_url)
    events = events_response.json()

    corner_event_id = None
    for event in events:
        if event.get('type', {}).get('name') == 'Pass' and event.get('pass', {}).get('type', {}).get('name') == 'Corner':
            corner_event_id = event.get('id')
            print(f"Found corner kick event: {corner_event_id}")
            print(f"Player: {event.get('player', {}).get('name')}")
            print(f"Team: {event.get('team', {}).get('name')}")
            break

    # Find corresponding freeze frame
    if corner_event_id:
        found = False
        for freeze_frame in data:
            if freeze_frame.get('event_uuid') == corner_event_id:
                found = True
                print(f"\n✓ Found freeze frame for corner")
                print(f"Number of visible players: {len(freeze_frame.get('freeze_frame', []))}")

                if len(freeze_frame.get('freeze_frame', [])) > 0:
                    first_player = freeze_frame['freeze_frame'][0]
                    print(f"\nFirst player structure:")
                    print(json.dumps(first_player, indent=2))
                break

        if not found:
            print(f"\n✗ No freeze frame found for corner event {corner_event_id}")
            # Just show first freeze frame as example
            if len(data) > 0 and len(data[0].get('freeze_frame', [])) > 0:
                print(f"\nShowing first available freeze frame as example:")
                first_player = data[0]['freeze_frame'][0]
                print(json.dumps(first_player, indent=2))

                # Check all fields across first few players
                print(f"\nChecking all unique fields in first 10 players:")
                all_fields = set()
                for player in data[0]['freeze_frame'][:10]:
                    all_fields.update(player.keys())
                print(f"All fields found: {sorted(all_fields)}")
else:
    print(f"✗ 360 data not found (status: {response.status_code})")
    print("Trying different match...")

    # Try a different match
    match_id = 3788741  # Another Champions League match
    url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Found {len(data)} freeze frames for match {match_id}")

        if len(data) > 0 and len(data[0].get('freeze_frame', [])) > 0:
            first_player = data[0]['freeze_frame'][0]
            print(f"\nFirst player structure:")
            print(json.dumps(first_player, indent=2))
