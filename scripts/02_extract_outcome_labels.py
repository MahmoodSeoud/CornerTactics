#!/usr/bin/env python3
"""
Extract outcome labels for corner kicks based on the immediate next event.

This script:
1. Loads corners_with_freeze_frames.json from Task 1
2. For each corner, finds the next event in the match event sequence
3. Maps the next event type to one of 4 outcome classes
4. Saves corners_with_labels.json with outcome field added
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Outcome mapping as specified in PLAN.md
OUTCOME_MAPPING = {
    "Ball Receipt*": "Ball Receipt",
    "Clearance": "Clearance",
    "Goal Keeper": "Goalkeeper",
    # Everything else → "Other"
    "Duel": "Other",
    "Pressure": "Other",
    "Pass": "Other",
    "Foul Committed": "Other",
    "Ball Recovery": "Other",
    "Block": "Other",
    "Interception": "Other",
    "Dispossessed": "Other",
    "Shot": "Other"
}


def map_event_to_outcome(event: Dict) -> str:
    """Map event type to outcome class.

    Args:
        event: Event dictionary with 'type' field

    Returns:
        Outcome class: 'Ball Receipt', 'Clearance', 'Goalkeeper', or 'Other'
    """
    event_type = event.get("type", {}).get("name", "")
    return OUTCOME_MAPPING.get(event_type, "Other")


def find_event_index(events: List[Dict], event_uuid: str) -> Optional[int]:
    """Find index of event with given UUID in event list.

    Args:
        events: List of event dictionaries
        event_uuid: UUID to search for

    Returns:
        Index of event, or None if not found
    """
    for i, event in enumerate(events):
        if event.get("id") == event_uuid:
            return i
    return None


def get_next_event(events: List[Dict], corner_index: int) -> Optional[Dict]:
    """Get the immediate next event after a corner.

    Args:
        events: List of event dictionaries
        corner_index: Index of corner event

    Returns:
        Next event dictionary, or None if corner is last event
    """
    if corner_index >= len(events) - 1:
        return None
    return events[corner_index + 1]


def load_corners(corners_file: Path) -> List[Dict]:
    """Load corners_with_freeze_frames.json file.

    Args:
        corners_file: Path to corners JSON file

    Returns:
        List of corner dictionaries
    """
    with open(corners_file, 'r') as f:
        return json.load(f)


def load_match_events(events_dir: Path, match_id: str) -> List[Dict]:
    """Load match events file for given match_id.

    Args:
        events_dir: Directory containing event JSON files
        match_id: Match ID to load

    Returns:
        List of event dictionaries
    """
    match_file = events_dir / f"{match_id}.json"
    with open(match_file, 'r') as f:
        return json.load(f)


def add_outcome_label(corner: Dict, match_events: List[Dict]) -> Dict:
    """Add outcome label to corner based on next event.

    Args:
        corner: Corner dictionary with 'event' field
        match_events: List of all events in the match

    Returns:
        Corner dictionary with 'outcome' field added
    """
    corner_uuid = corner["event"]["id"]
    corner_index = find_event_index(match_events, corner_uuid)

    if corner_index is None:
        corner["outcome"] = "Unknown"
        return corner

    next_event = get_next_event(match_events, corner_index)

    if next_event is None:
        corner["outcome"] = "Unknown"
        return corner

    corner["outcome"] = map_event_to_outcome(next_event)
    return corner


def save_labeled_corners(corners: List[Dict], output_file: Path):
    """Save corners with labels to output file.

    Args:
        corners: List of corner dictionaries with 'outcome' field
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(corners, f, indent=2)


def validate_distribution(corners: List[Dict]) -> Dict[str, float]:
    """Calculate outcome class distribution.

    Args:
        corners: List of corner dictionaries with 'outcome' field

    Returns:
        Dictionary mapping outcome classes to percentage
    """
    total = len(corners)
    counts = {}

    for corner in corners:
        outcome = corner.get("outcome", "Unknown")
        counts[outcome] = counts.get(outcome, 0) + 1

    # Convert to percentages
    distribution = {
        outcome: (count / total) * 100
        for outcome, count in counts.items()
    }

    return distribution


def main():
    """Main execution function."""
    # Setup paths
    data_root = Path(__file__).parent.parent / "data"
    processed_dir = data_root / "processed"
    events_dir = data_root / "statsbomb" / "events" / "events"

    # Load corners from Task 1
    corners_file = processed_dir / "corners_with_freeze_frames.json"
    print(f"Loading corners from {corners_file}...")
    corners = load_corners(corners_file)
    print(f"Loaded {len(corners)} corners")

    # Process each corner
    labeled_corners = []
    missing_events = 0

    for i, corner in enumerate(corners):
        if (i + 1) % 100 == 0:
            print(f"Processing corner {i + 1}/{len(corners)}...")

        match_id = corner["match_id"]

        try:
            # Load match events
            match_events = load_match_events(events_dir, match_id)

            # Add outcome label
            labeled = add_outcome_label(corner, match_events)
            labeled_corners.append(labeled)

        except FileNotFoundError:
            print(f"Warning: Events file not found for match {match_id}")
            corner["outcome"] = "Unknown"
            labeled_corners.append(corner)
            missing_events += 1
        except Exception as e:
            print(f"Error processing corner {i}: {e}")
            corner["outcome"] = "Unknown"
            labeled_corners.append(corner)

    # Save labeled corners
    output_file = processed_dir / "corners_with_labels.json"
    print(f"\nSaving labeled corners to {output_file}...")
    save_labeled_corners(labeled_corners, output_file)

    # Validate distribution
    print("\n=== Outcome Class Distribution ===")
    distribution = validate_distribution(labeled_corners)

    for outcome in ["Ball Receipt", "Clearance", "Goalkeeper", "Other"]:
        pct = distribution.get(outcome, 0)
        count = sum(1 for c in labeled_corners if c.get("outcome") == outcome)
        print(f"{outcome:15} {count:5} ({pct:5.1f}%)")

    if "Unknown" in distribution:
        count = sum(1 for c in labeled_corners if c.get("outcome") == "Unknown")
        print(f"{'Unknown':15} {count:5} ({distribution['Unknown']:5.1f}%)")

    print(f"\nTotal: {len(labeled_corners)}")
    print(f"Missing event files: {missing_events}")

    print("\n✓ Task 2 complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
