#!/usr/bin/env python3
"""
Phase 1 Demo: Load DFL tracking data, count corners, and visualize.

This script demonstrates the Phase 1 deliverables:
1. Load tracking and event data from all 7 DFL matches
2. Count corners across all matches
3. Extract one corner sequence with velocities
4. Save a visualization to results/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dfl import (
    load_tracking_data,
    load_event_data,
    find_corner_events,
    extract_corner_sequence,
    compute_velocities,
    plot_corner_frame,
)


def count_corners_all_matches(data_dir: Path) -> dict:
    """Count corners across all DFL matches."""
    # Find all unique match IDs
    position_files = sorted(data_dir.glob("*positions_raw*.xml"))
    match_ids = [f.stem.split("_")[-1] for f in position_files]

    results = {
        "matches": {},
        "total_corners": 0,
    }

    print(f"Found {len(match_ids)} matches in {data_dir}")
    print("=" * 60)

    for match_id in match_ids:
        print(f"\nProcessing match: {match_id}")

        try:
            # Load events for this specific match
            events = load_event_data("dfl", data_dir, match_id=match_id)
            corners = find_corner_events(events)

            results["matches"][match_id] = len(corners)
            results["total_corners"] += len(corners)

            print(f"  Corners found: {len(corners)}")

            # Print corner details
            for i, c in enumerate(corners[:3]):
                print(f"    {i+1}. {c.timestamp} by {c.team}")
            if len(corners) > 3:
                print(f"    ... and {len(corners) - 3} more")

        except Exception as e:
            print(f"  Error: {e}")
            results["matches"][match_id] = 0

    return results


def extract_and_visualize_corner(data_dir: Path, output_dir: Path):
    """Extract one corner and save visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Loading tracking and event data for visualization...")

    # Load first match
    tracking = load_tracking_data("dfl", data_dir)
    events = load_event_data("dfl", data_dir)
    corners = find_corner_events(events)

    if not corners:
        print("No corners found for visualization")
        return None

    print(f"Using corner at {corners[0].timestamp}")

    # Extract sequence
    frames = extract_corner_sequence(
        tracking, corners[0], pre_seconds=2.0, post_seconds=6.0
    )
    print(f"Extracted {len(frames)} frames")

    # Compute velocities
    velocities = compute_velocities(frames, fps=25)
    print(f"Computed velocities for {len(velocities)} frames")

    # Visualize frame at delivery (middle of sequence)
    mid_idx = len(frames) // 2
    output_path = output_dir / "corner_visualization.png"

    fig = plot_corner_frame(
        frame=frames[mid_idx],
        velocities=velocities[mid_idx],
        corner_event=corners[0],
        title=f"Corner Kick - {corners[0].team} at {corners[0].timestamp}",
        save_path=output_path,
    )

    print(f"\nVisualization saved to: {output_path}")

    import matplotlib.pyplot as plt

    plt.close(fig)

    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1: DFL Data Loading & Validation Demo")
    print("=" * 60)

    dfl_data_dir = project_root / "data" / "dfl"
    output_dir = project_root / "results" / "phase1"

    # Check data exists
    if not dfl_data_dir.exists():
        print(f"Error: DFL data directory not found: {dfl_data_dir}")
        return 1

    # Count corners
    results = count_corners_all_matches(dfl_data_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total matches: {len(results['matches'])}")
    print(f"Total corners: {results['total_corners']}")
    print(f"Corners per match: {results['total_corners'] / max(1, len(results['matches'])):.1f}")
    print()
    print("Per-match breakdown:")
    for match_id, count in results["matches"].items():
        print(f"  {match_id}: {count} corners")

    # Visualize
    viz_path = extract_and_visualize_corner(dfl_data_dir, output_dir)

    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)
    print(f"Visualization: {viz_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
