#!/usr/bin/env python3
"""Extract corner kick events from SoccerNet Labels-v2.json files.

Output: CSV with columns [corner_id, game_path, half, timestamp_seconds, position_ms, team, visibility]
"""

import json
import csv
import re
from pathlib import Path
from tqdm import tqdm


def parse_game_time(game_time: str) -> tuple[int, int]:
    """Parse gameTime string like '1 - 16:46' into (half, seconds).

    Returns:
        (half, total_seconds) where half is 1 or 2
    """
    match = re.match(r'(\d+)\s*-\s*(\d+):(\d+)', game_time)
    if not match:
        raise ValueError(f"Cannot parse game time: {game_time}")

    half = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))

    return half, minutes * 60 + seconds


def extract_corners_from_file(labels_path: Path) -> list[dict]:
    """Extract corner events from a single Labels-v2.json file."""
    with open(labels_path) as f:
        data = json.load(f)

    game_path = data.get('UrlLocal', str(labels_path.parent))
    corners = []

    for ann in data.get('annotations', []):
        if ann.get('label') == 'Corner':
            half, timestamp_sec = parse_game_time(ann['gameTime'])
            corners.append({
                'game_path': game_path,
                'half': half,
                'timestamp_seconds': timestamp_sec,
                'position_ms': int(ann.get('position', 0)),
                'team': ann.get('team', 'unknown'),
                'visibility': ann.get('visibility', 'unknown')
            })

    return corners


def main():
    # Find all Labels-v2.json files
    soccernet_dir = Path('/home/mseo/CornerTactics/data/misc/soccernet/videos')
    labels_files = list(soccernet_dir.rglob('Labels-v2.json'))

    print(f"Found {len(labels_files)} Labels-v2.json files")

    all_corners = []
    corner_id = 0

    for labels_path in tqdm(labels_files, desc="Processing games"):
        try:
            corners = extract_corners_from_file(labels_path)
            for corner in corners:
                corner['corner_id'] = corner_id
                all_corners.append(corner)
                corner_id += 1
        except Exception as e:
            print(f"Error processing {labels_path}: {e}")

    # Save to CSV
    output_dir = Path('/home/mseo/CornerTactics/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'soccernet_corners.csv'

    fieldnames = ['corner_id', 'game_path', 'half', 'timestamp_seconds', 'position_ms', 'team', 'visibility']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_corners)

    print(f"\nExtracted {len(all_corners)} corners from {len(labels_files)} games")
    print(f"Saved to: {output_path}")

    # Print statistics
    visible_corners = sum(1 for c in all_corners if c['visibility'] == 'visible')
    print(f"\nStatistics:")
    print(f"  Total corners: {len(all_corners)}")
    print(f"  Visible corners: {visible_corners}")
    print(f"  Not shown: {len(all_corners) - visible_corners}")
    print(f"  First half: {sum(1 for c in all_corners if c['half'] == 1)}")
    print(f"  Second half: {sum(1 for c in all_corners if c['half'] == 2)}")


if __name__ == '__main__':
    main()
