#!/usr/bin/env python3
"""
TASK 1: Data Extraction Pipeline

Extract corner kicks from StatsBomb event data and match with freeze frames.
Creates data/processed/corners_with_freeze_frames.json.

Usage:
    python scripts/01_extract_corners_with_freeze_frames.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


# File paths
EVENTS_DIR = Path("data/statsbomb/events/events")
FREEZE_FRAMES_DIR = Path("data/statsbomb/freeze-frames")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "corners_with_freeze_frames.json"


def is_corner_kick(event: Dict[str, Any]) -> bool:
    """
    Check if an event is a corner kick.

    Args:
        event: Event dictionary from StatsBomb data

    Returns:
        True if event is a corner kick, False otherwise
    """
    return (
        event.get("type", {}).get("name") == "Pass"
        and event.get("pass", {}).get("type", {}).get("name") == "Corner"
    )


def extract_corners_from_match(match_id: str) -> List[Dict[str, Any]]:
    """
    Extract all corner kicks from a match.

    Args:
        match_id: Match ID (filename without .json extension)

    Returns:
        List of corner kick events
    """
    event_file = EVENTS_DIR / f"{match_id}.json"

    if not event_file.exists():
        return []

    with open(event_file) as f:
        events = json.load(f)

    corners = [event for event in events if is_corner_kick(event)]

    return corners


def match_corners_with_freeze_frames(
    corners: List[Dict[str, Any]],
    freeze_frames: List[Dict[str, Any]],
    match_id: str,
) -> List[Dict[str, Any]]:
    """
    Match corner events with their freeze frames.

    Args:
        corners: List of corner kick events
        freeze_frames: List of freeze frame objects
        match_id: Match ID

    Returns:
        List of matched corner-freeze frame pairs
    """
    # Create lookup dict for faster matching
    freeze_frame_lookup = {ff["event_uuid"]: ff for ff in freeze_frames}

    matched_corners = []

    for corner in corners:
        corner_uuid = corner["id"]

        if corner_uuid in freeze_frame_lookup:
            matched_corners.append(
                {
                    "match_id": match_id,
                    "event": corner,
                    "freeze_frame": freeze_frame_lookup[corner_uuid]["freeze_frame"],
                }
            )

    return matched_corners


def process_all_matches(
    output_path: Optional[str] = None, limit: Optional[int] = None
) -> Dict[str, int]:
    """
    Process all matches and extract corners with freeze frames.

    Args:
        output_path: Path to save output file (default: OUTPUT_FILE)
        limit: Limit number of matches to process (for testing)

    Returns:
        Dictionary with processing statistics
    """
    if output_path is None:
        output_path = OUTPUT_FILE

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all match files
    match_files = sorted(EVENTS_DIR.glob("*.json"))

    if limit:
        match_files = match_files[:limit]

    all_corners_with_freeze_frames = []
    stats = {
        "total_matches_processed": 0,
        "total_corners_found": 0,
        "total_corners_with_freeze_frames": 0,
        "matches_with_freeze_frames": 0,
    }

    print(f"Processing {len(match_files)} matches...")

    for match_file in tqdm(match_files, desc="Extracting corners"):
        match_id = match_file.stem
        stats["total_matches_processed"] += 1

        # Extract corners from this match
        corners = extract_corners_from_match(match_id)
        stats["total_corners_found"] += len(corners)

        if not corners:
            continue

        # Check if freeze frame file exists
        freeze_frame_file = FREEZE_FRAMES_DIR / f"{match_id}.json"

        if not freeze_frame_file.exists():
            continue

        stats["matches_with_freeze_frames"] += 1

        # Load freeze frames
        with open(freeze_frame_file) as f:
            freeze_frames = json.load(f)

        # Match corners with freeze frames
        matched = match_corners_with_freeze_frames(corners, freeze_frames, match_id)
        all_corners_with_freeze_frames.extend(matched)
        stats["total_corners_with_freeze_frames"] += len(matched)

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_corners_with_freeze_frames, f, indent=2)

    return stats


def main():
    """Main execution function."""
    print("=" * 60)
    print("TASK 1: Extract Corners with Freeze Frames")
    print("=" * 60)

    stats = process_all_matches()

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total matches processed: {stats['total_matches_processed']}")
    print(f"Matches with freeze frames: {stats['matches_with_freeze_frames']}")
    print(f"Total corners found: {stats['total_corners_found']}")
    print(
        f"Corners with freeze frames: {stats['total_corners_with_freeze_frames']}"
    )
    print(f"\nOutput saved to: {OUTPUT_FILE}")

    # Validate against expected count
    expected_count = 1933
    actual_count = stats["total_corners_with_freeze_frames"]

    if actual_count >= expected_count * 0.9:  # Within 10% is acceptable
        print(f"\n✓ Output count ({actual_count}) matches expected (~{expected_count})")
    else:
        print(
            f"\n⚠ Warning: Output count ({actual_count}) is significantly lower than expected (~{expected_count})"
        )


if __name__ == "__main__":
    main()
