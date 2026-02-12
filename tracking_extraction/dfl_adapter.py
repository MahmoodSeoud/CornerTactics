"""DFL Bundesliga tracking data adapter.

Wraps existing src/dfl/ code to output the unified CornerTrackingData format.
DFL provides professional tracking data at 25Hz.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .core import (
    CornerTrackingData,
    Frame,
    PlayerFrame,
    compute_velocities_central_diff,
    normalize_to_pitch,
)

logger = logging.getLogger(__name__)

FPS = 25.0


def convert_dfl_match(
    tracking_dataset,
    event_dataset,
    match_id: str,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> List[CornerTrackingData]:
    """Convert DFL tracking data to unified format.

    Uses existing functions from src.dfl.data_loading and
    src.dfl.graph_construction to find corners, extract sequences,
    and label outcomes.

    Args:
        tracking_dataset: kloppy TrackingDataset (from DFL/Sportec loader)
        event_dataset: kloppy EventDataset
        match_id: Match identifier string (e.g. "DFL-MAT-J03WMX")
        pre_seconds: Seconds before corner delivery
        post_seconds: Seconds after corner delivery

    Returns:
        List of CornerTrackingData objects
    """
    from src.dfl.data_loading import (
        find_corner_events,
        extract_corner_sequence,
        compute_velocities,
    )
    from src.dfl.graph_construction import label_corner

    corners_events = find_corner_events(event_dataset)
    logger.info("Match %s: found %d corner events", match_id, len(corners_events))

    results = []
    for idx, corner_event in enumerate(corners_events):
        # Extract tracking frames around the corner
        kloppy_frames = extract_corner_sequence(
            tracking_dataset, corner_event,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )
        if not kloppy_frames:
            logger.warning("  Corner %d: no frames extracted, skipping", idx)
            continue

        # Get labels from existing code
        labels = label_corner(corner_event, event_dataset)

        # Determine outcome
        if labels["shot_binary"] == 1:
            outcome = "shot"
        else:
            outcome = "no_shot"

        # Identify attacking team
        attacking_team = getattr(corner_event, "team", None)

        # Convert kloppy frames to our Frame/PlayerFrame format
        frames = []
        for fi, kf in enumerate(kloppy_frames):
            players = []
            ball_x, ball_y = None, None

            for player_obj, player_data in kf.players_data.items():
                if player_data.coordinates is None:
                    continue

                pid = str(getattr(player_obj, "player_id", player_obj))
                raw_x = player_data.coordinates.x
                raw_y = player_data.coordinates.y

                # DFL coordinates via kloppy are already in meters (0-105, 0-68)
                x_m, y_m = normalize_to_pitch(raw_x, raw_y, "dfl")

                # Determine team
                team_str = "unknown"
                if attacking_team is not None and hasattr(player_obj, "team"):
                    if player_obj.team == attacking_team:
                        team_str = "attacking"
                    else:
                        team_str = "defending"

                # Determine role
                role_str = "player"
                pos = getattr(player_obj, "starting_position", None)
                if pos is not None:
                    pos_name = getattr(pos, "name", "")
                    if "goalkeeper" in str(pos_name).lower():
                        role_str = "goalkeeper"

                players.append(PlayerFrame(
                    player_id=pid,
                    team=team_str,
                    role=role_str,
                    x=x_m,
                    y=y_m,
                    is_visible=True,
                ))

            # Ball
            if kf.ball_coordinates is not None:
                bx, by = normalize_to_pitch(
                    kf.ball_coordinates.x,
                    kf.ball_coordinates.y,
                    "dfl",
                )
                ball_x, ball_y = bx, by

            if not players:
                continue

            timestamp_ms = _get_timestamp_ms(kf)
            frames.append(Frame(
                frame_idx=fi,
                timestamp_ms=timestamp_ms,
                players=players,
                ball_x=ball_x,
                ball_y=ball_y,
            ))

        if len(frames) < 5:
            logger.warning("  Corner %d: only %d frames, skipping", idx, len(frames))
            continue

        # Compute velocities
        compute_velocities_central_diff(frames, fps=FPS)

        # Determine delivery frame index (the frame closest to the corner event time)
        delivery_idx = _find_delivery_frame(frames, len(frames), pre_seconds)

        corner_time = _get_event_timestamp_s(corner_event)

        corner_data = CornerTrackingData(
            corner_id=f"dfl_{match_id}_corner_{idx}",
            source="dfl",
            match_id=match_id,
            delivery_frame=delivery_idx,
            fps=FPS,
            outcome=outcome,
            frames=frames,
            metadata={
                "labels": labels,
                "corner_time_s": corner_time,
                "period": str(getattr(corner_event.period, "id", ""))
                         if hasattr(corner_event, "period") else "",
            },
        )
        results.append(corner_data)

        n_players = [len(f.players) for f in frames]
        logger.info(
            "  Corner %d: %d frames, %.1f avg players, outcome=%s",
            idx, len(frames),
            sum(n_players) / len(n_players),
            outcome,
        )

    return results


def convert_dfl_from_paths(
    data_dir: Path,
    match_id: str,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> List[CornerTrackingData]:
    """Convenience: load DFL data from paths and convert.

    Args:
        data_dir: Directory containing DFL XML files
        match_id: Match identifier
        pre_seconds: Seconds before corner delivery
        post_seconds: Seconds after corner delivery
    """
    from src.dfl.data_loading import load_tracking_data, load_event_data

    tracking = load_tracking_data("dfl", data_dir, match_id=match_id)
    events = load_event_data("dfl", data_dir, match_id=match_id)

    return convert_dfl_match(
        tracking, events, match_id,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )


def _get_timestamp_ms(frame) -> float:
    """Extract timestamp in milliseconds from a kloppy frame."""
    ts = frame.timestamp
    if hasattr(ts, "total_seconds"):
        return ts.total_seconds() * 1000.0
    return float(ts) * 1000.0


def _get_event_timestamp_s(event) -> float:
    """Extract timestamp in seconds from a kloppy event."""
    ts = event.timestamp
    if hasattr(ts, "total_seconds"):
        return ts.total_seconds()
    return float(ts)


def _find_delivery_frame(frames: List[Frame], n_frames: int, pre_seconds: float) -> int:
    """Estimate delivery frame index from pre_seconds offset.

    The delivery should be at approximately pre_seconds into the window.
    """
    if not frames:
        return 0
    # The delivery is at time = pre_seconds from the start of the window.
    # Frame 0 is at the start, so delivery ≈ pre_seconds * fps.
    target_idx = int(pre_seconds * FPS)
    return min(target_idx, n_frames - 1)
