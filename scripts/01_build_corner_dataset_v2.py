#!/usr/bin/env python3
"""
Step 1 (v2): Build corner kick dataset using IMMEDIATE NEXT EVENT labeling.

This follows the TacticAI methodology - label based on what happens immediately
after the corner, not within a 15-second window.

Labels:
- SHOT: Goal, Shots on target, Shots off target
- NO_SHOT: Ball out of play, Foul, Offside, Clearance, etc.

Secondary labels for NO_SHOT breakdown:
- CLEARANCE: Ball out of play, Clearance
- FOUL: Foul
- OFFSIDE: Offside
- OTHER: Everything else

Usage:
    python scripts/01_build_corner_dataset_v2.py

Output:
    FAANTRA/data/corners/corner_dataset_v2.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime


# Default paths
DEFAULT_SOCCERNET_PATH = Path("data/misc/soccernet/videos")
DEFAULT_OUTPUT_PATH = Path("FAANTRA/data/corners")

# Configuration
MAX_TIME_TO_NEXT_EVENT_MS = 30000  # Only consider next event within 30s
CLIP_BEFORE_MS = 25000  # Observation window: 25s before corner
CLIP_AFTER_MS = 5000    # Anticipation window: 5s after corner
FPS = 25

# Event to label mapping
SHOT_EVENTS = {"Goal", "Shots on target", "Shots off target"}
NO_SHOT_EVENTS = {
    "Ball out of play": "CLEARANCE",
    "Clearance": "CLEARANCE",
    "Foul": "FOUL",
    "Direct free-kick": "FOUL",
    "Indirect free-kick": "FOUL",
    "Offside": "OFFSIDE",
    "Throw-in": "CLEARANCE",
    "Corner": "OTHER",  # Won another corner
    "Kick-off": "OTHER",  # Could be goal (handled separately) or half end
    "Substitution": "OTHER",
    "Yellow card": "FOUL",
    "Red card": "FOUL",
}


def classify_corner_outcome(corner_event: dict, next_event: dict) -> dict:
    """
    Classify corner outcome based on immediate next event.

    Returns:
        dict with 'primary' (SHOT/NO_SHOT) and 'secondary' label
    """
    if next_event is None:
        return {"primary": "NO_SHOT", "secondary": "OTHER", "next_event": "none"}

    next_label = next_event.get("label", "")
    time_diff = int(next_event["position"]) - int(corner_event["position"])

    # Too far away - probably unrelated
    if time_diff > MAX_TIME_TO_NEXT_EVENT_MS or time_diff < 0:
        return {"primary": "NO_SHOT", "secondary": "OTHER", "next_event": "timeout"}

    # Check if it's a shot
    if next_label in SHOT_EVENTS:
        if next_label == "Goal":
            secondary = "GOAL"
        elif next_label == "Shots on target":
            secondary = "ON_TARGET"
        else:
            secondary = "OFF_TARGET"
        return {
            "primary": "SHOT",
            "secondary": secondary,
            "next_event": next_label,
            "time_to_next_ms": time_diff
        }

    # It's a no-shot outcome
    secondary = NO_SHOT_EVENTS.get(next_label, "OTHER")
    return {
        "primary": "NO_SHOT",
        "secondary": secondary,
        "next_event": next_label,
        "time_to_next_ms": time_diff
    }


def find_corners_in_match(labels_file: Path, soccernet_path: Path) -> list:
    """
    Find all corner kicks in a single match with immediate next event labeling.
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

    # Sort by game time (half, then position within half)
    def sort_key(x):
        gt = x.get("gameTime", "0 - 00:00")
        half = int(gt[0]) if gt[0].isdigit() else 0
        pos = int(x.get("position", 0))
        return (half, pos)

    annotations = sorted(annotations, key=sort_key)

    corners = []

    for i, ann in enumerate(annotations):
        if ann.get("label") != "Corner":
            continue

        corner_time = int(ann["position"])
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

        # Find immediate next event (same half)
        next_event = None
        for j in range(i + 1, len(annotations)):
            candidate = annotations[j]
            candidate_time = candidate.get("gameTime", "")

            # Must be same half
            if not candidate_time.startswith(str(half)):
                break

            # Skip non-meaningful events
            if candidate.get("label") in {"Substitution", "Yellow card", "Red card"}:
                continue

            next_event = candidate
            break

        # Classify outcome
        outcome = classify_corner_outcome(ann, next_event)

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
            "primary_label": outcome["primary"],
            "secondary_label": outcome["secondary"],
            "next_event": outcome.get("next_event", ""),
            "time_to_next_ms": outcome.get("time_to_next_ms", -1),
            "visibility": ann.get("visibility", "visible"),
            "team": ann.get("team", "unknown"),
            "game_time": game_time
        })

    return corners


def build_dataset(soccernet_path: Path, output_path: Path) -> dict:
    """Build the complete corner kick dataset with immediate next event labeling."""
    print(f"Scanning SoccerNet matches in: {soccernet_path}")

    all_corners = []
    primary_counts = Counter()
    secondary_counts = Counter()
    next_event_counts = Counter()
    match_count = 0

    # Find all matches with labels
    labels_files = sorted(soccernet_path.glob("**/Labels-v2.json"))
    print(f"Found {len(labels_files)} matches with labels")

    for labels_file in labels_files:
        corners = find_corners_in_match(labels_file, soccernet_path)
        if corners:
            match_count += 1
            for corner in corners:
                primary_counts[corner["primary_label"]] += 1
                secondary_counts[corner["secondary_label"]] += 1
                next_event_counts[corner["next_event"]] += 1
            all_corners.extend(corners)

    print(f"\nProcessed {match_count} matches with videos")
    print(f"Total corners found: {len(all_corners)}")

    # Print primary distribution (SHOT vs NO_SHOT)
    print(f"\n=== Primary Labels (SHOT vs NO_SHOT) ===")
    for label, count in primary_counts.most_common():
        pct = count / len(all_corners) * 100 if all_corners else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Print secondary distribution
    print(f"\n=== Secondary Labels ===")
    for label, count in secondary_counts.most_common():
        pct = count / len(all_corners) * 100 if all_corners else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Print next event distribution
    print(f"\n=== Next Event Distribution ===")
    for label, count in next_event_counts.most_common(10):
        pct = count / len(all_corners) * 100 if all_corners else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Create dataset structure
    dataset = {
        "version": "2.0",
        "labeling_method": "immediate_next_event",
        "created": datetime.now().isoformat(),
        "config": {
            "max_time_to_next_event_ms": MAX_TIME_TO_NEXT_EVENT_MS,
            "clip_before_ms": CLIP_BEFORE_MS,
            "clip_after_ms": CLIP_AFTER_MS,
            "clip_duration_ms": CLIP_BEFORE_MS + CLIP_AFTER_MS,
            "fps": FPS
        },
        "primary_classes": ["SHOT", "NO_SHOT"],
        "secondary_classes": {
            "SHOT": ["GOAL", "ON_TARGET", "OFF_TARGET"],
            "NO_SHOT": ["CLEARANCE", "FOUL", "OFFSIDE", "OTHER"]
        },
        "statistics": {
            "total_corners": len(all_corners),
            "matches_with_corners": match_count,
            "primary_distribution": dict(primary_counts),
            "secondary_distribution": dict(secondary_counts),
            "next_event_distribution": dict(next_event_counts)
        },
        "corners": all_corners
    }

    # Save dataset
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "corner_dataset_v2.json"

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to: {output_file}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build corner kick dataset with immediate next event labeling"
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
