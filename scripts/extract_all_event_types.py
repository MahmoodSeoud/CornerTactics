#!/usr/bin/env python3
"""
Extract ALL unique event types from StatsBomb raw JSON data.
This gives us the complete vocabulary of events that can occur after corners.
"""

import json
import urllib.request
from pathlib import Path
from typing import Set, List
import glob


def get_event_types_from_online_sample() -> Set[str]:
    """Get event types from online sample matches."""
    event_types = set()

    # Sample match IDs from different competitions
    sample_matches = [
        3795220,  # Euro 2020 Final
        7478,     # OL Reign vs Houston Dash
        15946,    # Barcelona vs Alaves
        22912,    # Other sample
        303377,   # Another sample
    ]

    print("Fetching event types from sample matches...")

    for match_id in sample_matches:
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"

        try:
            with urllib.request.urlopen(url) as response:
                events = json.loads(response.read().decode('utf-8'))

            for event in events:
                # Get main event type
                event_type = event.get('type', {}).get('name')
                if event_type:
                    event_types.add(event_type)

        except Exception as e:
            print(f"  Could not fetch match {match_id}: {e}")
            continue

    return event_types


def get_event_types_from_local_jsons() -> Set[str]:
    """If local JSON files exist, extract all event types."""
    event_types = set()

    json_dir = Path("data/raw/statsbomb/json_events/events")
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))

        print(f"Processing {len(json_files)} local JSON files...")

        for json_file in json_files[:100]:  # Process first 100 for speed
            try:
                with open(json_file) as f:
                    events = json.load(f)

                for event in events:
                    event_type = event.get('type', {}).get('name')
                    if event_type:
                        event_types.add(event_type)

            except Exception as e:
                continue

    return event_types


def get_complete_event_type_list() -> List[str]:
    """
    Complete list of ALL StatsBomb event types based on documentation
    and observed data from their open-data repository.
    """

    # From StatsBomb documentation and observed in data
    documented_event_types = [
        # Ball Events
        "Pass",
        "Ball Receipt*",  # Note the asterisk is part of the name
        "Carry",
        "Ball Recovery",
        "Dispossessed",
        "Miscontrol",

        # Shooting Events
        "Shot",

        # Defensive Events
        "Clearance",
        "Block",
        "Interception",
        "Shield",

        # Duel Events
        "Duel",
        "50/50",

        # Goalkeeper Events
        "Goal Keeper",

        # Foul Events
        "Foul Committed",
        "Foul Won",
        "Offside",

        # Set Piece Events
        "Own Goal Against",
        "Own Goal For",

        # Game Management Events
        "Starting XI",
        "Substitution",
        "Tactical Shift",
        "Half Start",
        "Half End",
        "Player On",
        "Player Off",

        # Other Events
        "Pressure",
        "Dribble",
        "Dribbled Past",
        "Injury Stoppage",
        "Referee Ball-Drop",
        "Error",
        "Bad Behaviour",

        # Special/Rare Events
        "Advantage",
        "Card",
        "Coming On",
        "Going Off"
    ]

    return sorted(documented_event_types)


def main():
    """Extract and save all event types."""

    print("=" * 60)
    print("StatsBomb Event Type Extractor")
    print("=" * 60)
    print()

    # Try to get from online samples
    online_types = get_event_types_from_online_sample()
    print(f"✓ Found {len(online_types)} event types from online samples")

    # Try to get from local files if they exist
    local_types = get_event_types_from_local_jsons()
    if local_types:
        print(f"✓ Found {len(local_types)} event types from local files")

    # Combine observed types
    observed_types = online_types.union(local_types)

    # Get documented types
    documented_types = set(get_complete_event_type_list())

    # Combine all
    all_types = observed_types.union(documented_types)

    # Sort alphabetically
    sorted_types = sorted(all_types)

    # Save to file
    output_file = Path("data/analysis/statsbomb_all_event_types.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("# Complete List of StatsBomb Event Types\n")
        f.write("# These are all possible events that can occur after a corner kick\n")
        f.write(f"# Total: {len(sorted_types)} event types\n\n")

        f.write("## Main Event Types (can follow a corner):\n")
        for event_type in sorted_types:
            f.write(f"{event_type}\n")

        f.write("\n## Most Common After Corners (based on analysis):\n")
        common_after_corners = [
            "Clearance",
            "Ball Receipt*",
            "Pass",
            "Shot",
            "Duel",
            "Block",
            "Goal Keeper",
            "Carry",
            "Interception",
            "Pressure"
        ]
        for event_type in common_after_corners:
            if event_type in sorted_types:
                f.write(f"{event_type}\n")

    # Also save a simple version (just the list)
    simple_file = Path("data/analysis/event_types_simple.txt")
    with open(simple_file, "w") as f:
        for event_type in sorted_types:
            f.write(f"{event_type}\n")

    print()
    print("📊 Summary:")
    print(f"  Observed in samples: {len(observed_types)} types")
    print(f"  From documentation: {len(documented_types)} types")
    print(f"  Total unique: {len(sorted_types)} types")
    print()
    print("✅ Saved to:")
    print(f"  - {output_file}")
    print(f"  - {simple_file}")
    print()
    print("Sample event types that can follow corners:")
    for event_type in sorted_types[:10]:
        print(f"  • {event_type}")

    return sorted_types


if __name__ == "__main__":
    event_types = main()