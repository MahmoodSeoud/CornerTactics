#!/usr/bin/env python3
"""
Extract binary shot labels for corner kicks.

Following TacticAI methodology:
- Look ahead at next 5 events after corner kick
- Check for THREATENING shots from ATTACKING team only
- Threatening shots: Goal, Saved, Post, Off Target, Wayward
- Exclude: Blocked shots, shots from defending team

This script:
1. Loads corners_with_freeze_frames.json from Task 1
2. For each corner, looks ahead at next 5 events
3. Checks if any event is a threatening Shot from attacking team
4. Assigns binary label: 1 (Shot) or 0 (No Shot)
5. Saves corners_with_shot_labels.json with shot_outcome field added
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


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


def check_shot_in_lookahead(
    events: List[Dict],
    corner_index: int,
    window_size: int,
    attacking_team_id: int
) -> bool:
    """Check if any event in lookahead window is a threatening Shot by attacking team.

    Following TacticAI methodology:
    - Only count shots from the attacking team (corner-taking team)
    - Only count "threatening shots": Goal, Saved, Post, Off Target, Wayward
    - Exclude blocked shots and shots from defending team

    Args:
        events: List of event dictionaries
        corner_index: Index of corner event
        window_size: Number of events to look ahead
        attacking_team_id: ID of corner-taking team

    Returns:
        True if threatening shot found in window, False otherwise
    """
    # Threatening shot outcomes (following TacticAI)
    THREATENING_OUTCOMES = {
        "Goal",           # Direct goal
        "Saved",          # Shot saved by goalkeeper
        "Post",           # Hit the post/crossbar
        "Off T",          # Off target (but clear attempt)
        "Wayward",        # Missing target (but clear attempt)
    }

    # Calculate end of lookahead window
    start_idx = corner_index + 1
    end_idx = min(start_idx + window_size, len(events))

    # Check each event in window
    for i in range(start_idx, end_idx):
        event = events[i]
        event_type = event.get("type", {}).get("name", "")

        # Check if it's a Shot event
        if event_type == "Shot":
            # Check if shot is from attacking team
            shot_team_id = event.get("team", {}).get("id")
            if shot_team_id != attacking_team_id:
                continue  # Skip shots from defending team

            # Check if shot outcome is threatening
            shot_outcome = event.get("shot", {}).get("outcome", {}).get("name", "")
            if shot_outcome in THREATENING_OUTCOMES:
                return True

    return False


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


def add_shot_label(
    corner: Dict,
    match_events: List[Dict],
    window_size: int = 5
) -> Dict:
    """Add binary shot label to corner.

    Args:
        corner: Corner dictionary with 'event' field
        match_events: List of all events in the match
        window_size: Number of events to look ahead (default: 5)

    Returns:
        Corner dictionary with 'shot_outcome' field added
    """
    corner_uuid = corner["event"]["id"]
    corner_index = find_event_index(match_events, corner_uuid)

    if corner_index is None:
        corner["shot_outcome"] = 0  # Default to no shot
        return corner

    # Get attacking team ID from corner event
    attacking_team_id = corner["event"].get("team", {}).get("id")
    if attacking_team_id is None:
        corner["shot_outcome"] = 0
        return corner

    has_shot = check_shot_in_lookahead(
        match_events, corner_index, window_size, attacking_team_id
    )
    corner["shot_outcome"] = 1 if has_shot else 0

    return corner


def save_labeled_corners(corners: List[Dict], output_file: Path):
    """Save corners with shot labels to output file.

    Args:
        corners: List of corner dictionaries with 'shot_outcome' field
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(corners, f, indent=2)


def validate_distribution(corners: List[Dict]) -> Dict[str, float]:
    """Calculate shot vs no-shot class distribution.

    Args:
        corners: List of corner dictionaries with 'shot_outcome' field

    Returns:
        Dictionary with shot_percentage, no_shot_percentage, and imbalance_ratio
    """
    total = len(corners)
    if total == 0:
        return {
            "shot_percentage": 0.0,
            "no_shot_percentage": 0.0,
            "imbalance_ratio": 0.0
        }

    shot_count = sum(1 for c in corners if c.get("shot_outcome") == 1)
    no_shot_count = total - shot_count

    shot_pct = (shot_count / total) * 100
    no_shot_pct = (no_shot_count / total) * 100

    # Calculate imbalance ratio (no_shot / shot)
    imbalance = no_shot_count / shot_count if shot_count > 0 else 0.0

    return {
        "shot_percentage": shot_pct,
        "no_shot_percentage": no_shot_pct,
        "imbalance_ratio": imbalance,
        "shot_count": shot_count,
        "no_shot_count": no_shot_count,
        "total": total
    }


def main():
    """Main execution function."""
    # Setup paths
    data_root = Path(__file__).parent.parent / "data"
    processed_dir = data_root / "processed"
    events_dir = data_root / "statsbomb" / "events" / "events"

    # Configuration
    WINDOW_SIZE = 5  # Following TacticAI paper

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

            # Add shot label
            labeled = add_shot_label(corner, match_events, WINDOW_SIZE)
            labeled_corners.append(labeled)

        except FileNotFoundError:
            print(f"Warning: Events file not found for match {match_id}")
            corner["shot_outcome"] = 0
            labeled_corners.append(corner)
            missing_events += 1
        except Exception as e:
            print(f"Error processing corner {i}: {e}")
            corner["shot_outcome"] = 0
            labeled_corners.append(corner)

    # Save labeled corners
    output_file = processed_dir / "corners_with_shot_labels.json"
    print(f"\nSaving labeled corners to {output_file}...")
    save_labeled_corners(labeled_corners, output_file)

    # Validate distribution
    print("\n=== Binary Shot Class Distribution ===")
    distribution = validate_distribution(labeled_corners)

    print(f"Shot:        {distribution['shot_count']:5} ({distribution['shot_percentage']:5.1f}%)")
    print(f"No Shot:     {distribution['no_shot_count']:5} ({distribution['no_shot_percentage']:5.1f}%)")
    print(f"\nTotal:       {distribution['total']}")
    print(f"Imbalance:   {distribution['imbalance_ratio']:.2f}:1")
    print(f"Missing event files: {missing_events}")

    # Validate expected range (TacticAI reported ~24%)
    shot_pct = distribution['shot_percentage']
    if shot_pct < 15:
        print(f"\n⚠️  WARNING: Shot percentage ({shot_pct:.1f}%) is below expected range (15-30%)")
        print("    Consider increasing lookahead window size")
    elif shot_pct > 35:
        print(f"\n⚠️  WARNING: Shot percentage ({shot_pct:.1f}%) is above expected range (15-30%)")
        print("    Consider decreasing lookahead window size")
    else:
        print(f"\n✓ Shot percentage ({shot_pct:.1f}%) is within expected range (15-30%)")

    print("\n✓ Task 7 complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
