#!/usr/bin/env python3
"""
Quick test to demonstrate the corner transition analysis.
Shows P(a_{t+1} | corner_t) using real StatsBomb data.
"""

import json
import urllib.request
from collections import defaultdict


def quick_corner_analysis():
    """Quick demo of corner transition analysis."""

    print("=" * 60)
    print("CORNER KICK TRANSITION ANALYSIS - Quick Demo")
    print("P(a_{t+1} | corner_t)")
    print("=" * 60)

    # Download a sample match
    print("\n📥 Downloading sample match data...")

    # Get Euro 2020 Final (has good data)
    match_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3795220.json"

    try:
        with urllib.request.urlopen(match_url) as response:
            events = json.loads(response.read().decode('utf-8'))
        print(f"✓ Loaded {len(events)} events")
    except Exception as e:
        print(f"Failed to download sample match: {e}")
        return

    # Find corners and analyze transitions
    corner_transitions = defaultdict(int)
    corner_details = []

    for i, event in enumerate(events):
        # Check if it's a corner
        if event.get('type', {}).get('name') == 'Pass':
            pass_obj = event.get('pass', {})
            if pass_obj.get('type', {}).get('name') == 'Corner':
                # Found a corner!
                corner_team = event.get('team', {}).get('name')
                corner_player = event.get('player', {}).get('name')
                corner_minute = event.get('minute')

                # Look at next event
                if i + 1 < len(events):
                    next_event = events[i + 1]
                    next_type = next_event.get('type', {}).get('name', 'Unknown')

                    # Add context for certain events
                    if next_type == 'Shot':
                        shot_outcome = next_event.get('shot', {}).get('outcome', {}).get('name', '')
                        next_description = f"Shot({shot_outcome})"
                    elif next_type == 'Pass':
                        pass_outcome = next_event.get('pass', {}).get('outcome', {}).get('name', 'Complete')
                        next_description = f"Pass({pass_outcome})"
                    elif next_type == 'Clearance':
                        next_description = "Clearance"
                    else:
                        next_description = next_type

                    corner_transitions[next_description] += 1

                    corner_details.append({
                        'minute': corner_minute,
                        'team': corner_team,
                        'player': corner_player,
                        'next_event': next_description
                    })

    # Print results
    print(f"\n🎯 Found {len(corner_details)} corners in this match")

    print("\n📊 TRANSITION PROBABILITIES: P(a_{t+1} | corner_t)")
    print("-" * 40)

    total = sum(corner_transitions.values())
    for event, count in sorted(corner_transitions.items(), key=lambda x: x[1], reverse=True):
        prob = count / total if total > 0 else 0
        print(f"  {event:20s}: {prob:6.1%} ({count} times)")

    print("\n🔍 CORNER DETAILS:")
    print("-" * 40)
    for corner in corner_details:
        print(f"  {corner['minute']}' - {corner['team']} ({corner['player']})")
        print(f"       → {corner['next_event']}")

    print("\n💡 INSIGHTS:")
    print("-" * 40)

    # Calculate key metrics
    shots = sum(1 for c in corner_details if 'Shot' in c['next_event'])
    clearances = sum(1 for c in corner_details if 'Clearance' in c['next_event'])

    print(f"  • Shots after corner: {shots}/{len(corner_details)} ({shots/len(corner_details)*100:.0f}%)")
    print(f"  • Cleared immediately: {clearances}/{len(corner_details)} ({clearances/len(corner_details)*100:.0f}%)")

    print("\n✅ Demo complete! This shows the basic analysis.")
    print("   For full dataset, run: python scripts/download_statsbomb_raw_jsons.py")
    print("   Then: python scripts/analyze_corner_transitions.py")


if __name__ == "__main__":
    quick_corner_analysis()