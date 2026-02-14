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


def parse_gsr_output(gsr_json_path: Path) -> List[Frame]:
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

    Timestamps are left as 0.0 — the caller should set them once the
    actual GSR output fps is known (see gsr_to_corner_tracking).

    Args:
        gsr_json_path: Path to GSR JSON output file

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

        # Quality filter: skip frames with very few players
        # GSR on broadcast video may detect fewer players in wide shots
        if len(players) < 5:
            continue

        frames.append(Frame(
            frame_idx=fid,
            timestamp_ms=0.0,  # Set by caller once gsr_fps is known
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


def _detect_corner_taker_team(
    frames: List[Frame],
) -> Optional[str]:
    """Detect which GSR team label ("left"/"right") is the corner-taking (attacking) team.

    Searches all frames for the player closest to any corner flag, since the
    corner taker approaches the flag before delivery but moves away after.
    Corner flags are at (0, 0), (0, 68), (105, 0), (105, 68) in pitch coordinates.

    Args:
        frames: All frames from the clip (searched globally)

    Returns:
        "left" or "right" (the GSR team label of the attacking team), or None if unknown
    """
    if not frames:
        return None

    corner_flags = [
        (0.0, 0.0), (0.0, PITCH_WIDTH),
        (PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH),
    ]

    best_dist = float("inf")
    best_team = None

    # Search all frames for the player closest to any corner flag
    for frame in frames:
        for player in frame.players:
            if player.team not in ("left", "right"):
                continue
            for fx, fy in corner_flags:
                dist = math.sqrt((player.x - fx) ** 2 + (player.y - fy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_team = player.team

    # Only trust if the nearest player is within ~10 meters of a corner flag.
    # Broadcast tracking has positional noise, so we use a generous threshold.
    if best_dist > 10.0:
        return None

    return best_team


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
    pre_seconds: float = 10.0,
    post_seconds: float = 0.0,
    clip_duration_s: float = 30.0,
) -> Optional[CornerTrackingData]:
    """Convert GSR output for a single clip to CornerTrackingData.

    FAANTRA clips are 30s of pre-delivery observation, so the corner delivery
    is at the end of the clip. GSR subsamples frames (~250 output frames for
    a 750-frame 25fps input), so frame indices are sequential output indices.

    Args:
        gsr_output_path: Path to GSR JSON output for this clip
        corner_metadata: Dict from prepare_gsr_clip_list() entry
        pre_seconds: Seconds before delivery to include
        post_seconds: Seconds after delivery to include
        clip_duration_s: Duration of source clip in seconds

    Returns:
        CornerTrackingData or None if insufficient quality
    """
    # Parse GSR output (timestamps set below once gsr_fps is known)
    all_frames = parse_gsr_output(gsr_output_path)
    if not all_frames:
        logger.warning("No valid frames in GSR output: %s", gsr_output_path)
        return None

    # Detect actual GSR output frame rate from the data.
    # GSR subsamples: ~250 output frames for a 30s clip ≈ 8.33 fps.
    max_frame_idx = max(f.frame_idx for f in all_frames)
    gsr_fps = (max_frame_idx + 1) / clip_duration_s

    # Delivery is at the end of the clip (clips are pre-delivery observation)
    delivery_frame = max_frame_idx

    # Fix timestamps to use GSR output fps
    for f in all_frames:
        f.timestamp_ms = (f.frame_idx / gsr_fps) * 1000.0

    # Extract window: pre_seconds before delivery, post_seconds after
    pre_frames = int(pre_seconds * gsr_fps)
    post_frames = int(post_seconds * gsr_fps)
    start_frame = delivery_frame - pre_frames
    end_frame = delivery_frame + post_frames

    window_frames = [
        f for f in all_frames
        if start_frame <= f.frame_idx <= end_frame
    ]

    min_frames = 20  # Lower threshold since GSR is ~8 fps
    if len(window_frames) < min_frames:
        logger.warning(
            "Corner %s: only %d frames in window (need >=%d)",
            corner_metadata["corner_id"], len(window_frames), min_frames,
        )
        return None

    # Filter wild position jumps (threshold scaled for GSR fps)
    window_frames = _filter_position_jumps(
        window_frames, max_jump_m=5.0, fps=gsr_fps,
    )

    if len(window_frames) < min_frames:
        logger.warning(
            "Corner %s: only %d frames after jump filter",
            corner_metadata["corner_id"], len(window_frames),
        )
        return None

    # Resolve team labels from "left"/"right" to "attacking"/"defending"
    # Search ALL frames (not just window) since the corner taker is near
    # the flag earlier in the clip, before the observation window starts.
    corner_team_side = _detect_corner_taker_team(all_frames)
    if corner_team_side:
        for frame in window_frames:
            _resolve_teams(frame.players, corner_team_side)
    else:
        logger.debug("Could not resolve team sides for %s", corner_metadata["corner_id"])

    # Compute velocities using GSR output fps
    compute_velocities_central_diff(window_frames, fps=gsr_fps)

    # Map outcome from corner_dataset.json
    raw_outcome = corner_metadata.get("outcome", "")
    if raw_outcome in SHOT_OUTCOMES:
        outcome = "shot"
    else:
        outcome = "no_shot"

    # Find delivery frame index within window
    delivery_idx = len(window_frames) - 1  # Delivery is at end of clip

    corner_id = f"gsr_{corner_metadata['corner_id']}"

    return CornerTrackingData(
        corner_id=corner_id,
        source="soccernet_gsr",
        match_id=corner_metadata.get("match_dir", ""),
        delivery_frame=delivery_idx,
        fps=gsr_fps,
        outcome=outcome,
        frames=window_frames,
        metadata={
            "raw_outcome": raw_outcome,
            "corner_time_ms": corner_metadata["corner_time_ms"],
            "clip_start_ms": corner_metadata["clip_start_ms"],
            "gsr_output": str(gsr_output_path),
            "gsr_fps": gsr_fps,
        },
    )


def process_gsr_outputs(
    gsr_output_dir: Path,
    clip_list_path: Path,
    pre_seconds: float = 10.0,
    post_seconds: float = 0.0,
) -> List[CornerTrackingData]:
    """Process all GSR outputs from a batch run.

    Expects GSR output JSON files named {corner_id}.json in gsr_output_dir.

    Args:
        gsr_output_dir: Directory containing GSR JSON outputs
        clip_list_path: Path to clip list from prepare_gsr_clip_list()
        pre_seconds: Seconds before delivery
        post_seconds: Seconds after delivery

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
        )
        if corner_data is not None:
            results.append(corner_data)

    logger.info("Processed %d / %d GSR outputs successfully", len(results), len(clip_list))
    return results
