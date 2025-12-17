#!/usr/bin/env python3
"""
Step 1: Build corner kick dataset from SoccerNet-v2 labels.

This script scans SoccerNet match directories for corner kick events and creates
a dataset JSON file with metadata for each corner including:
- Match information
- Corner timing
- Outcome classification (GOAL, SHOT, CLEARED, etc.)

Usage:
    python scripts/01_build_corner_dataset.py [--soccernet-path PATH] [--output-path PATH]

Output:
    data/corners/corner_dataset.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


# Default paths (relative to CornerTactics root)
DEFAULT_SOCCERNET_PATH = Path("data/misc/soccernet/videos")
DEFAULT_OUTPUT_PATH = Path("FAANTRA/data/corners")

# Configuration
OUTCOME_WINDOW_MS = 15000  # Look for outcome events within 15s after corner
CLIP_BEFORE_MS = 25000     # Observation window: 25s before corner
CLIP_AFTER_MS = 5000       # Anticipation window: 5s after corner
FPS = 25

# Outcome classes (ordered by "danger" level for classification priority)
OUTCOME_PRIORITY = [
    ("Goal", "GOAL"),
    ("Penalty", "GOAL"),  # Merge penalty into GOAL
    ("Shots on target", "SHOT_ON_TARGET"),
    ("Shots off target", "SHOT_OFF_TARGET"),
    ("Offside", "OFFSIDE"),
    ("Foul", "FOUL"),
    ("Direct free-kick", "FOUL"),
    ("Indirect free-kick", "FOUL"),
    ("Corner", "CORNER_WON"),
    ("Clearance", "CLEARED"),
    ("Ball out of play", "CLEARED"),
    ("Throw-in", "CLEARED"),
]


def classify_outcome(events: list) -> str:
    """
    Classify corner outcome based on subsequent events.

    Args:
        events: List of event dictionaries from SoccerNet labels

    Returns:
        Outcome class string
    """
    if not events:
        return "NOT_DANGEROUS"

    labels = {e["label"] for e in events}

    # Check priority order
    for soccernet_label, outcome_class in OUTCOME_PRIORITY:
        if soccernet_label in labels:
            return outcome_class

    return "NOT_DANGEROUS"


def find_corners_in_match(labels_file: Path, soccernet_path: Path) -> list:
    """
    Find all corner kicks in a single match.

    Args:
        labels_file: Path to Labels-v2.json
        soccernet_path: Root path of SoccerNet videos

    Returns:
        List of corner dictionaries
    """
    match_dir = labels_file.parent

    # Check for video files
    half1_video = match_dir / "1_720p.mkv"
    half2_video = match_dir / "2_720p.mkv"

    if not half1_video.exists() and not half2_video.exists():
        return []

    with open(labels_file) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    corners = []

    for ann in annotations:
        if ann.get("label") != "Corner":
            continue

        corner_time = int(ann["position"])  # milliseconds
        game_time = ann.get("gameTime", "")

        # Determine which half
        if game_time.startswith("1"):
            half = 1
            video_file = half1_video
        else:
            half = 2
            video_file = half2_video

        if not video_file.exists():
            continue

        # Find events within outcome window (same half only)
        outcome_events = [
            a for a in annotations
            if corner_time < int(a["position"]) <= corner_time + OUTCOME_WINDOW_MS
            and a.get("gameTime", "").startswith(str(half))
            and a.get("label") != "Corner"  # Exclude the corner itself
        ]

        outcome = classify_outcome(outcome_events)

        # Calculate clip timing
        clip_start_ms = max(0, corner_time - CLIP_BEFORE_MS)
        clip_end_ms = corner_time + CLIP_AFTER_MS

        corners.append({
            "match_dir": str(match_dir.relative_to(soccernet_path)),
            "video_file": str(video_file),
            "half": half,
            "corner_time_ms": corner_time,
            "clip_start_ms": clip_start_ms,
            "clip_end_ms": clip_end_ms,
            "outcome": outcome,
            "outcome_events": [e["label"] for e in outcome_events],
            "visibility": ann.get("visibility", "visible"),
            "team": ann.get("team", "unknown"),
            "game_time": game_time
        })

    return corners


def build_dataset(soccernet_path: Path, output_path: Path) -> dict:
    """
    Build the complete corner kick dataset.

    Args:
        soccernet_path: Root path to SoccerNet videos
        output_path: Output directory for dataset files

    Returns:
        Dataset dictionary
    """
    print(f"Scanning SoccerNet matches in: {soccernet_path}")

    all_corners = []
    outcome_counts = Counter()
    match_count = 0

    # Find all matches with labels
    labels_files = sorted(soccernet_path.glob("**/Labels-v2.json"))
    print(f"Found {len(labels_files)} matches with labels")

    for labels_file in labels_files:
        corners = find_corners_in_match(labels_file, soccernet_path)
        if corners:
            match_count += 1
            for corner in corners:
                outcome_counts[corner["outcome"]] += 1
            all_corners.extend(corners)

    print(f"\nProcessed {match_count} matches with videos")
    print(f"Total corners found: {len(all_corners)}")

    # Print outcome distribution
    print(f"\nOutcome distribution:")
    for outcome, count in outcome_counts.most_common():
        pct = count / len(all_corners) * 100 if all_corners else 0
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # Create dataset structure
    dataset = {
        "version": "2.0",
        "created": datetime.now().isoformat(),
        "config": {
            "outcome_window_ms": OUTCOME_WINDOW_MS,
            "clip_before_ms": CLIP_BEFORE_MS,
            "clip_after_ms": CLIP_AFTER_MS,
            "clip_duration_ms": CLIP_BEFORE_MS + CLIP_AFTER_MS,
            "fps": FPS
        },
        "outcome_classes": [
            "GOAL",
            "SHOT_ON_TARGET",
            "SHOT_OFF_TARGET",
            "CORNER_WON",
            "CLEARED",
            "NOT_DANGEROUS",
            "FOUL",
            "OFFSIDE"
        ],
        "statistics": {
            "total_corners": len(all_corners),
            "matches_with_corners": match_count,
            "outcome_distribution": dict(outcome_counts)
        },
        "corners": all_corners
    }

    # Save dataset
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "corner_dataset.json"

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to: {output_file}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build corner kick dataset from SoccerNet labels"
    )
    parser.add_argument(
        "--soccernet-path",
        type=Path,
        default=DEFAULT_SOCCERNET_PATH,
        help="Path to SoccerNet videos directory"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory for dataset files"
    )
    args = parser.parse_args()

    if not args.soccernet_path.exists():
        print(f"Error: SoccerNet path does not exist: {args.soccernet_path}")
        sys.exit(1)

    build_dataset(args.soccernet_path, args.output_path)


if __name__ == "__main__":
    main()
