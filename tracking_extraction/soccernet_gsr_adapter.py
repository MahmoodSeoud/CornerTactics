"""SoccerNet Game State Recognition (GSR) adapter.

Parses sn-gamestate pipeline output and converts to unified CornerTrackingData.
GSR runs on existing 30s corner clips from FAANTRA/data/corners/clips/ and
produces per-frame player detections with pitch coordinates.

Typical workflow:
1. Prepare clip list from corner_dataset.json
2. Run GSR pipeline via SLURM (separate step)
3. Parse GSR JSON output -> CornerTrackingData
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from .core import (
    CornerTrackingData,
    Frame,
    PlayerFrame,
    compute_velocities_central_diff,
    normalize_to_pitch,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)

# GSR output FPS matches the input video FPS
DEFAULT_FPS = 25.0

# Outcome mapping from corner_dataset.json to binary
SHOT_OUTCOMES = {"SHOT_ON_TARGET", "SHOT_OFF_TARGET", "GOAL"}


def parse_gsr_output(gsr_json_path: Path, fps: float = DEFAULT_FPS) -> List[Frame]:
    """Parse sn-gamestate JSON predictions into Frame objects.

    GSR output is a list of detections (COCO-style), each with:
        {
            "image_id": frame_id (int),
            "track_id": int,
            "attributes": {
                "role": "player" | "goalkeeper" | "referee" | "other",
                "team": "left" | "right",
                "jersey": int
            },
            "bbox_pitch": {
                "x_bottom_middle": float,
                "y_bottom_middle": float
            }
        }

    Ball detections have role="ball".

    Args:
        gsr_json_path: Path to GSR JSON output file
        fps: Frame rate of the source video

    Returns:
        List of Frame objects sorted by frame_idx
    """
    with open(gsr_json_path) as f:
        detections = json.load(f)

    # Group detections by image_id (frame)
    by_frame: Dict[int, list] = defaultdict(list)
    for det in detections:
        fid = det.get("image_id", det.get("frame_id"))
        if fid is not None:
            by_frame[int(fid)].append(det)

    frames = []
    for fid in sorted(by_frame.keys()):
        dets = by_frame[fid]
        players = []
        ball_x, ball_y = None, None

        for det in dets:
            attrs = det.get("attributes", {})
            role = attrs.get("role", "player")

            # Get pitch coordinates
            bbox_pitch = det.get("bbox_pitch", {})
            x_raw = bbox_pitch.get("x_bottom_middle")
            y_raw = bbox_pitch.get("y_bottom_middle")

            if x_raw is None or y_raw is None:
                continue

            if role == "ball":
                bx, by = normalize_to_pitch(x_raw, y_raw, "soccernet_gsr")
                ball_x, ball_y = bx, by
                continue

            if role == "referee":
                continue

            # Normalize coordinates
            x_m, y_m = normalize_to_pitch(x_raw, y_raw, "soccernet_gsr")

            # Team mapping: "left"/"right" -> resolved later
            team_raw = attrs.get("team", "unknown")
            track_id = det.get("track_id", det.get("id", 0))

            player_role = "goalkeeper" if role == "goalkeeper" else "player"

            players.append(PlayerFrame(
                player_id=f"track_{track_id}",
                team=team_raw,  # Will be resolved to attacking/defending later
                role=player_role,
                x=x_m,
                y=y_m,
                is_visible=True,
            ))

        # Quality filter: skip frames with too few players
        if len(players) < 10:
            continue

        frames.append(Frame(
            frame_idx=fid,
            timestamp_ms=(fid / fps) * 1000.0,
            players=players,
            ball_x=ball_x,
            ball_y=ball_y,
        ))

    return frames


def prepare_gsr_clip_list(
    corner_dataset_json: Path,
    clips_dir: Path,
    output_list: Path,
    max_corners: Optional[int] = None,
) -> List[Dict]:
    """Generate list of corner clips for GSR processing.

    Filters to visible corners only and creates a manifest for the GSR
    SLURM pipeline.

    Args:
        corner_dataset_json: Path to FAANTRA/data/corners/corner_dataset.json
        clips_dir: Path to FAANTRA/data/corners/clips/
        output_list: Path to write the clip list JSON
        max_corners: Maximum number of corners to include (None = all)

    Returns:
        List of dicts with corner metadata
    """
    with open(corner_dataset_json) as f:
        dataset = json.load(f)

    clips = []
    for i, corner in enumerate(dataset["corners"]):
        if corner.get("visibility") != "visible":
            continue

        corner_id = f"corner_{i:04d}"
        clip_path = Path(clips_dir) / corner_id / "720p.mp4"

        if not clip_path.exists():
            logger.debug("Clip not found: %s", clip_path)
            continue

        clips.append({
            "corner_id": corner_id,
            "corner_idx": i,
            "clip_path": str(clip_path),
            "corner_time_ms": corner["corner_time_ms"],
            "clip_start_ms": corner["clip_start_ms"],
            "outcome": corner["outcome"],
            "match_dir": corner["match_dir"],
        })

        if max_corners and len(clips) >= max_corners:
            break

    # Save clip list
    output_list = Path(output_list)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    with open(output_list, "w") as f:
        json.dump(clips, f, indent=2)

    logger.info("Prepared %d clips for GSR processing -> %s", len(clips), output_list)
    return clips


def _resolve_teams(
    players: List[PlayerFrame],
    corner_team_side: str,
) -> List[PlayerFrame]:
    """Map GSR team labels ("left"/"right") to "attacking"/"defending".

    The corner taker's side determines which team is attacking.

    Args:
        players: List of PlayerFrame with team="left"|"right"
        corner_team_side: Which side the corner-taking team is on ("left" or "right")

    Returns:
        Same players with team updated to "attacking"/"defending"
    """
    for p in players:
        if p.team == corner_team_side:
            p.team = "attacking"
        elif p.team in ("left", "right"):
            p.team = "defending"
        else:
            p.team = "unknown"
    return players


def _filter_position_jumps(
    frames: List[Frame],
    max_jump_m: float = 5.0,
    fps: float = DEFAULT_FPS,
) -> List[Frame]:
    """Remove frames where any player has an implausible position jump.

    GSR detections can occasionally produce wild position swaps or
    tracking errors. This filters out such frames.

    Args:
        frames: Sorted list of frames
        max_jump_m: Maximum allowed position change per frame (meters)
        fps: Frame rate

    Returns:
        Filtered list of frames
    """
    if len(frames) < 2:
        return frames

    # Build position history per track_id
    prev_pos: Dict[str, tuple] = {}
    filtered = [frames[0]]

    for pf in frames[0].players:
        prev_pos[pf.player_id] = (pf.x, pf.y)

    for frame in frames[1:]:
        has_wild_jump = False
        for pf in frame.players:
            if pf.player_id in prev_pos:
                px, py = prev_pos[pf.player_id]
                dx = pf.x - px
                dy = pf.y - py
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > max_jump_m:
                    has_wild_jump = True
                    break

        if not has_wild_jump:
            filtered.append(frame)
            for pf in frame.players:
                prev_pos[pf.player_id] = (pf.x, pf.y)

    return filtered


def gsr_to_corner_tracking(
    gsr_output_path: Path,
    corner_metadata: Dict[str, Any],
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
    fps: float = DEFAULT_FPS,
) -> Optional[CornerTrackingData]:
    """Convert GSR output for a single clip to CornerTrackingData.

    Args:
        gsr_output_path: Path to GSR JSON output for this clip
        corner_metadata: Dict from prepare_gsr_clip_list() entry
        pre_seconds: Seconds before delivery to include
        post_seconds: Seconds after delivery to include
        fps: Video frame rate

    Returns:
        CornerTrackingData or None if insufficient quality
    """
    # Parse GSR output
    all_frames = parse_gsr_output(gsr_output_path, fps=fps)
    if not all_frames:
        logger.warning("No valid frames in GSR output: %s", gsr_output_path)
        return None

    # Determine delivery frame from corner timing
    # Clips start at clip_start_ms, corner is at corner_time_ms
    clip_start_ms = corner_metadata["clip_start_ms"]
    corner_time_ms = corner_metadata["corner_time_ms"]
    delivery_time_in_clip_s = (corner_time_ms - clip_start_ms) / 1000.0
    delivery_frame = int(delivery_time_in_clip_s * fps)

    # Extract window around delivery
    pre_frames = int(pre_seconds * fps)
    post_frames = int(post_seconds * fps)
    start_frame = delivery_frame - pre_frames
    end_frame = delivery_frame + post_frames

    window_frames = [
        f for f in all_frames
        if start_frame <= f.frame_idx <= end_frame
    ]

    if len(window_frames) < 50:
        logger.warning(
            "Corner %s: only %d frames in window (need >=50)",
            corner_metadata["corner_id"], len(window_frames),
        )
        return None

    # Filter wild position jumps
    window_frames = _filter_position_jumps(window_frames, max_jump_m=5.0, fps=fps)

    if len(window_frames) < 50:
        logger.warning(
            "Corner %s: only %d frames after jump filter",
            corner_metadata["corner_id"], len(window_frames),
        )
        return None

    # Resolve team labels from "left"/"right" to "attacking"/"defending"
    # Corner taker's team can be inferred from the corner_dataset.json
    # For now, use "left"/"right" as-is since we don't know which side
    # the corner-taking team is on without additional match context.
    # TODO: resolve once GSR pipeline provides camera orientation info

    # Compute velocities
    compute_velocities_central_diff(window_frames, fps=fps)

    # Map outcome from corner_dataset.json
    raw_outcome = corner_metadata.get("outcome", "")
    if raw_outcome in SHOT_OUTCOMES:
        outcome = "shot"
    else:
        outcome = "no_shot"

    # Find delivery frame index within window
    delivery_idx = 0
    for i, f in enumerate(window_frames):
        if f.frame_idx >= delivery_frame:
            delivery_idx = i
            break

    corner_id = f"gsr_{corner_metadata['corner_id']}"

    return CornerTrackingData(
        corner_id=corner_id,
        source="soccernet_gsr",
        match_id=corner_metadata.get("match_dir", ""),
        delivery_frame=delivery_idx,
        fps=fps,
        outcome=outcome,
        frames=window_frames,
        metadata={
            "raw_outcome": raw_outcome,
            "corner_time_ms": corner_time_ms,
            "clip_start_ms": clip_start_ms,
            "gsr_output": str(gsr_output_path),
        },
    )


def process_gsr_outputs(
    gsr_output_dir: Path,
    clip_list_path: Path,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
    fps: float = DEFAULT_FPS,
) -> List[CornerTrackingData]:
    """Process all GSR outputs from a batch run.

    Expects GSR output JSON files named {corner_id}.json in gsr_output_dir.

    Args:
        gsr_output_dir: Directory containing GSR JSON outputs
        clip_list_path: Path to clip list from prepare_gsr_clip_list()
        pre_seconds: Seconds before delivery
        post_seconds: Seconds after delivery
        fps: Video frame rate

    Returns:
        List of CornerTrackingData
    """
    with open(clip_list_path) as f:
        clip_list = json.load(f)

    results = []
    for meta in clip_list:
        cid = meta["corner_id"]
        gsr_path = Path(gsr_output_dir) / f"{cid}.json"

        if not gsr_path.exists():
            logger.debug("GSR output not found: %s", gsr_path)
            continue

        corner_data = gsr_to_corner_tracking(
            gsr_path, meta,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            fps=fps,
        )
        if corner_data is not None:
            results.append(corner_data)

    logger.info("Processed %d / %d GSR outputs successfully", len(results), len(clip_list))
    return results
