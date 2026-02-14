#!/usr/bin/env python3
"""Extract per-corner JSON from TrackLab .pklz state files.

TrackLab saves detections as pickled DataFrames inside zipfiles (.pklz).
This script reads them and produces per-corner JSON files matching the
format expected by soccernet_gsr_adapter.py.

Supports two naming conventions:
1. Per-corner files: CORNER-XXXX.pklz (one video per file, video_id=0)
2. Batched files: batch_XXXX.pklz (multiple videos, video_id=corner stem)

Must run in the sn-gamestate venv (needs matching pandas pickle format).

Usage:
    cd /home/mseo/CornerTactics/sn-gamestate
    source .venv/bin/activate
    python ../tracking_extraction/scripts/extract_pklz_to_json.py \
        --pklz-dir ../outputs/gsr_states \
        --output-dir ../tracking_extraction/output/gsr_raw
"""

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_pklz(
    pklz_path: Path,
    output_dir: Path,
    corner_id_override: Optional[str] = None,
) -> List[str]:
    """Extract per-corner JSONs from a single .pklz state file.

    Args:
        pklz_path: Path to .pklz file
        output_dir: Where to write corner JSON files
        corner_id_override: If set, use this as the corner ID (for single-corner files)

    Returns:
        List of corner IDs successfully extracted
    """
    extracted = []

    try:
        with zipfile.ZipFile(pklz_path, "r") as zf:
            pkl_names = [
                n for n in zf.namelist()
                if n.endswith(".pkl") and "_image" not in n and n != "summary.json"
            ]

            for pkl_name in pkl_names:
                video_id = pkl_name.replace(".pkl", "")

                # Determine corner_id
                if corner_id_override:
                    corner_id = corner_id_override
                else:
                    corner_id = video_id

                try:
                    with zf.open(pkl_name) as fp:
                        detections_df = pd.read_pickle(fp)
                except Exception as e:
                    logger.warning("Failed to load %s from %s: %s", pkl_name, pklz_path, e)
                    continue

                if detections_df.empty:
                    logger.warning("Empty detections for %s", corner_id)
                    continue

                json_detections = dataframe_to_json(detections_df)

                if not json_detections:
                    logger.warning("No valid detections for %s", corner_id)
                    continue

                output_path = output_dir / f"{corner_id}.json"
                with open(output_path, "w") as f:
                    json.dump(json_detections, f)

                n_frames = len(set(d["image_id"] for d in json_detections))
                extracted.append(corner_id)
                logger.debug(
                    "%s: %d detections, %d frames",
                    corner_id, len(json_detections), n_frames,
                )
    except zipfile.BadZipFile:
        logger.warning("Bad zip file: %s", pklz_path)
    except Exception as e:
        logger.warning("Failed to process %s: %s", pklz_path, e)

    return extracted


def dataframe_to_json(df: pd.DataFrame) -> List[Dict]:
    """Convert a TrackLab detections DataFrame to adapter JSON format.

    Expected columns: image_id, track_id, bbox_pitch (dict), role_detection/role, team
    """
    detections = []

    for _, row in df.iterrows():
        bbox_pitch = row.get("bbox_pitch")
        if bbox_pitch is None or (isinstance(bbox_pitch, float) and np.isnan(bbox_pitch)):
            continue

        if isinstance(bbox_pitch, dict):
            x = bbox_pitch.get("x_bottom_middle")
            y = bbox_pitch.get("y_bottom_middle")
        else:
            continue

        if x is None or y is None:
            continue

        # Get role — check both "role" (aggregated) and "role_detection" (per-frame)
        role = row.get("role", row.get("role_detection", "player"))
        if isinstance(role, (int, float)):
            role = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}.get(
                int(role), "other"
            )
        if pd.isna(role) if isinstance(role, float) else False:
            role = "player"

        # Get team
        team = row.get("team", "unknown")
        if isinstance(team, float) and pd.isna(team):
            team = "unknown"

        # Get track_id
        track_id = row.get("track_id", row.name)
        if isinstance(track_id, float):
            track_id = int(track_id) if not pd.isna(track_id) else int(row.name)

        # Get image_id (frame index)
        image_id = row.get("image_id")
        if isinstance(image_id, float):
            image_id = int(image_id)

        detections.append({
            "image_id": int(image_id),
            "track_id": int(track_id),
            "attributes": {
                "role": str(role),
                "team": str(team),
            },
            "bbox_pitch": {
                "x_bottom_middle": float(x),
                "y_bottom_middle": float(y),
            },
        })

    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-corner JSON from TrackLab .pklz files"
    )
    parser.add_argument(
        "--pklz-dir",
        type=str,
        default="outputs/gsr_states",
        help="Directory containing .pklz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../tracking_extraction/output/gsr_raw",
        help="Output directory for per-corner JSON files",
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=None,
        help="Limit number of corners to extract",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pklz_dir = Path(args.pklz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect naming convention
    per_corner_files = sorted(pklz_dir.glob("CORNER-*.pklz"))
    batch_files = sorted(pklz_dir.glob("batch_*.pklz"))

    total_extracted = 0

    if per_corner_files:
        logger.info("Found %d per-corner pklz files", len(per_corner_files))
        files_to_process = per_corner_files
        if args.max_corners:
            files_to_process = files_to_process[:args.max_corners]

        for pklz_path in files_to_process:
            # CORNER-0123.pklz -> corner_0123
            match = re.match(r"CORNER-(\d+)", pklz_path.stem)
            if match:
                corner_id = f"corner_{int(match.group(1)):04d}"
            else:
                corner_id = pklz_path.stem.lower().replace("-", "_")

            extracted = extract_pklz(pklz_path, output_dir, corner_id_override=corner_id)
            total_extracted += len(extracted)

    elif batch_files:
        logger.info("Found %d batch pklz files", len(batch_files))
        for pklz_path in batch_files:
            extracted = extract_pklz(pklz_path, output_dir)
            total_extracted += len(extracted)
            if args.max_corners and total_extracted >= args.max_corners:
                break

    else:
        logger.error("No .pklz files found in %s", pklz_dir)
        return

    print(f"\nExtracted {total_extracted} corners")
    print(f"Output: {output_dir}")

    # Quick stats
    json_files = list(output_dir.glob("corner_*.json"))
    if json_files:
        sizes = [f.stat().st_size for f in json_files[:10]]
        print(f"JSON files: {len(json_files)}")
        print(f"Avg file size: {sum(sizes)/len(sizes)/1024:.1f} KB")


if __name__ == "__main__":
    main()
