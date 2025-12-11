#!/usr/bin/env python3
"""Format corner clips for sn-gamestate GSR pipeline.

Creates the SoccerNetGS directory structure expected by tracklab:
  MySoccerNetGS/
  └── custom/
      ├── CORNER-0000/
      │   ├── video.mp4
      │   └── Labels-GameState.json
      ├── CORNER-0001/
      │   └── ...

Each Labels-GameState.json contains minimal structure (no ground truth annotations).
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def create_minimal_labels_json() -> dict:
    """Create minimal Labels-GameState.json structure."""
    return {
        "info": {"version": "1.3"},
        "images": [],
        "annotations": []
    }


def main():
    parser = argparse.ArgumentParser(description='Format corner clips for GSR pipeline')
    parser.add_argument('--clips-dir', type=str,
                        default='/home/mseo/CornerTactics/data/corner_clips',
                        help='Directory containing corner_XXXX.mp4 clips')
    parser.add_argument('--output-dir', type=str,
                        default='/home/mseo/CornerTactics/data/MySoccerNetGS',
                        help='Output directory for GSR-formatted data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without creating files')
    parser.add_argument('--symlink', action='store_true', default=True,
                        help='Use symlinks instead of copying videos (default: True)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy videos instead of symlinking')
    args = parser.parse_args()

    if args.copy:
        args.symlink = False

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    custom_dir = output_dir / 'custom'

    # Find all corner clips
    clips = sorted(clips_dir.glob('corner_*.mp4'))
    print(f"Found {len(clips)} corner clips in {clips_dir}")

    if not clips:
        print("No clips found. Run extract_corner_clips.py first.")
        return

    if args.dry_run:
        print(f"\nDry run - would create:")
        print(f"  Output dir: {custom_dir}")
        for clip in clips[:5]:
            corner_id = clip.stem.replace('corner_', '')
            print(f"  - CORNER-{corner_id}/")
            print(f"      video.mp4 -> {clip}")
            print(f"      Labels-GameState.json")
        if len(clips) > 5:
            print(f"  ... and {len(clips) - 5} more")
        return

    # Create output directory
    custom_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure for each clip
    success = 0
    for clip in tqdm(clips, desc="Formatting for GSR"):
        # Extract corner ID from filename (corner_0000.mp4 -> 0000)
        corner_id = clip.stem.replace('corner_', '')
        corner_dir = custom_dir / f'CORNER-{corner_id}'

        # Create corner directory
        corner_dir.mkdir(exist_ok=True)

        # Link/copy video
        video_dest = corner_dir / 'video.mp4'
        if not video_dest.exists():
            if args.symlink:
                # Create relative symlink
                video_dest.symlink_to(clip.resolve())
            else:
                shutil.copy2(clip, video_dest)

        # Create Labels-GameState.json
        labels_path = corner_dir / 'Labels-GameState.json'
        if not labels_path.exists():
            with open(labels_path, 'w') as f:
                json.dump(create_minimal_labels_json(), f, indent=2)

        success += 1

    print(f"\nFormatted {success} corners for GSR pipeline")
    print(f"Output directory: {custom_dir}")

    # Create video list file for SLURM array indexing
    video_list_path = output_dir / 'video_list.txt'
    with open(video_list_path, 'w') as f:
        for clip in clips:
            corner_id = clip.stem.replace('corner_', '')
            f.write(f"CORNER-{corner_id}\n")

    print(f"Video list saved to: {video_list_path}")
    print(f"Total videos: {len(clips)}")


if __name__ == '__main__':
    main()
