"""SkillCorner open data adapter.

Extracts corner kick tracking data from SkillCorner's open dataset
(A-League broadcast tracking at 10Hz).

Data must be cloned locally first (tracking files use Git LFS):
    git clone https://github.com/SkillCorner/opendata.git <data_dir>
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from kloppy._providers.skillcorner import load as skillcorner_load

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

# All 10 available match IDs in the SkillCorner open dataset
ALL_MATCH_IDS = [
    2017461, 2015213, 2013725, 2011166, 2006229,
    1996435, 1953632, 1925299, 1899585, 1886347,
]

FPS = 10.0  # SkillCorner broadcast tracking rate


def load_match_tracking(match_id: int, data_dir: Path):
    """Load tracking data for a single match via kloppy.

    Args:
        match_id: SkillCorner match ID
        data_dir: Path to cloned SkillCorner opendata repo

    Returns:
        kloppy TrackingDataset
    """
    match_dir = Path(data_dir) / "data" / "matches" / str(match_id)
    meta_path = match_dir / f"{match_id}_match.json"
    raw_path = match_dir / f"{match_id}_tracking_extrapolated.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Match metadata not found: {meta_path}")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Tracking data not found: {raw_path}. "
            "Did you clone with git-lfs? Try: git lfs pull"
        )
    # Check if the file is an LFS pointer (small text file ~130 bytes)
    if raw_path.stat().st_size < 1000:
        with open(raw_path, "r") as f:
            first_line = f.readline()
        if first_line.startswith("version https://git-lfs"):
            raise FileNotFoundError(
                f"Tracking file is an LFS pointer (not downloaded): {raw_path}. "
                "Run: cd <data_dir> && git lfs pull"
            )

    return skillcorner_load(
        meta_data=str(meta_path),
        raw_data=str(raw_path),
        coordinates="skillcorner",  # Center-origin meters (±52, ±34)
    )


def load_match_metadata(match_id: int, data_dir: Path) -> dict:
    """Load match metadata JSON."""
    meta_path = Path(data_dir) / "data" / "matches" / str(match_id) / f"{match_id}_match.json"
    with open(meta_path) as f:
        return json.load(f)


def find_corners_from_events(match_id: int, data_dir: Path) -> List[Dict]:
    """Parse dynamic_events.csv for corner kick deliveries.

    Corner deliveries are identified by `game_interruption_before` containing
    'corner_for' or 'corner_against'. The first event after such an interruption
    is the corner delivery.

    Returns:
        List of dicts with keys:
            frame_start: int (frame number at 10Hz)
            period: int (1 or 2)
            team_id: int (team taking the corner)
            team_name: str
            attacking_side: str ("left_to_right" or "right_to_left")
            player_name: str (corner taker)
    """
    csv_path = (
        Path(data_dir) / "data" / "matches" / str(match_id)
        / f"{match_id}_dynamic_events.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {csv_path}")

    # Load match metadata for team ID mapping
    meta = load_match_metadata(match_id, data_dir)
    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    home_name = meta["home_team"]["name"]
    away_name = meta["away_team"]["name"]

    corners = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gi_before = row.get("game_interruption_before", "")
            if "corner" not in gi_before.lower():
                continue

            # corner_for = the team in this row takes the corner
            # corner_against = the opposing team takes it (this row's team defends)
            # In both cases, this is the first event after the corner restart
            team_id_str = row.get("team_id", "")
            team_name = row.get("team_shortname", "")

            # Determine which team is taking the corner
            if "corner_for" in gi_before:
                # This team is taking the corner
                corner_team_id = int(team_id_str) if team_id_str else None
                corner_team_name = team_name
            else:
                # corner_against: the OTHER team is taking the corner
                if team_id_str:
                    tid = int(team_id_str)
                    if tid == home_id:
                        corner_team_id = away_id
                        corner_team_name = away_name
                    else:
                        corner_team_id = home_id
                        corner_team_name = home_name
                else:
                    corner_team_id = None
                    corner_team_name = "unknown"

            corners.append({
                "frame_start": int(row["frame_start"]),
                "period": int(row["period"]),
                "team_id": corner_team_id,
                "team_name": corner_team_name,
                "attacking_side": row.get("attacking_side", ""),
                "player_name": row.get("player_name", ""),
                "gi_before": gi_before,
            })

    # Deduplicate: multiple events can share the same corner restart.
    # Keep the first event per unique (period, frame_start within 2 seconds) group.
    deduped = []
    dedup_window = int(2.0 * FPS)  # 20 frames at 10Hz
    for c in corners:
        is_dup = False
        for existing in deduped:
            if (existing["period"] == c["period"] and
                    abs(existing["frame_start"] - c["frame_start"]) <= dedup_window):
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)

    logger.info("Match %d: found %d corner events (%d after dedup)",
                match_id, len(corners), len(deduped))
    return deduped


def _determine_outcome_from_events(
    match_id: int,
    data_dir: Path,
    corner_frame: int,
    corner_period: int,
    post_window_frames: int = 150,  # 15 seconds at 10Hz
) -> Optional[str]:
    """Check post-corner events for shots.

    Looks at the `lead_to_shot` and `lead_to_goal` columns within the
    post-corner window. SkillCorner's event data doesn't have explicit
    'shot' event types, so we rely on these boolean columns.

    Returns:
        "shot" if any event in the window has lead_to_shot=True, else "no_shot"
    """
    csv_path = (
        Path(data_dir) / "data" / "matches" / str(match_id)
        / f"{match_id}_dynamic_events.csv"
    )
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["period"]) != corner_period:
                continue
            frame = int(row["frame_start"])
            if frame < corner_frame:
                continue
            if frame > corner_frame + post_window_frames:
                break

            if row.get("lead_to_shot", "").lower() == "true":
                return "shot"
            if row.get("lead_to_goal", "").lower() == "true":
                return "shot"

    return "no_shot"


def extract_corner_window(
    tracking,
    corner_event: Dict,
    match_metadata: dict,
    match_id: int,
    corner_idx: int,
    data_dir: Path,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> Optional[CornerTrackingData]:
    """Extract a 10s window (5s pre + 5s post) around corner delivery.

    Args:
        tracking: kloppy TrackingDataset
        corner_event: Dict from find_corners_from_events()
        match_metadata: Match JSON dict
        match_id: SkillCorner match ID
        corner_idx: Index of this corner within the match
        data_dir: Path to SkillCorner data repo
        pre_seconds: Seconds before delivery
        post_seconds: Seconds after delivery

    Returns:
        CornerTrackingData or None if insufficient data
    """
    corner_frame = corner_event["frame_start"]
    corner_period = corner_event["period"]
    corner_team_id = corner_event["team_id"]

    # Build player metadata lookup: player_id (str) -> {team_id, role}
    # kloppy uses the SkillCorner 'id' field as player_id (as string)
    player_lookup = {}
    for p in match_metadata.get("players", []):
        pid = p.get("id")
        if pid is not None:
            role_info = p.get("player_role", {})
            is_gk = "goalkeeper" in role_info.get("name", "").lower()
            player_lookup[str(pid)] = {
                "team_id": p.get("team_id"),
                "role": "goalkeeper" if is_gk else "player",
                "name": p.get("short_name", ""),
            }

    # Calculate frame window
    pre_frames = int(pre_seconds * FPS)
    post_frames = int(post_seconds * FPS)
    start_frame = corner_frame - pre_frames
    end_frame = corner_frame + post_frames

    # Extract frames from tracking dataset
    frames = []
    for record in tracking.records:
        # Match period
        if hasattr(record, "period"):
            period_num = getattr(record.period, "id", None)
            if period_num is not None:
                try:
                    if int(period_num) != corner_period:
                        continue
                except (ValueError, TypeError):
                    pass

        # Use frame_id for frame number
        fid = record.frame_id
        if fid is None:
            continue
        try:
            fid = int(fid)
        except (ValueError, TypeError):
            continue

        if fid < start_frame or fid > end_frame:
            continue

        # Convert kloppy record to our Frame
        players = []
        ball_x, ball_y = None, None

        for player_obj, player_data in record.players_data.items():
            if player_data.coordinates is None:
                continue

            # kloppy uses player objects as keys
            player_id_str = str(getattr(player_obj, "player_id", player_obj))

            # Get raw coordinates (SkillCorner: center-origin meters)
            raw_x = player_data.coordinates.x
            raw_y = player_data.coordinates.y

            # Normalize to standard pitch
            x_m, y_m = normalize_to_pitch(raw_x, raw_y, "skillcorner")

            # Determine team and role from match metadata lookup
            team_str = "unknown"
            role_str = "player"

            # Use metadata lookup for role (reliable GK detection)
            if player_id_str in player_lookup:
                meta_info = player_lookup[player_id_str]
                role_str = meta_info["role"]

            # Use kloppy's team info for team assignment
            if hasattr(player_obj, "team"):
                player_team = player_obj.team
                player_team_id = getattr(player_team, "team_id", None)
                if player_team_id is not None and corner_team_id is not None:
                    if str(player_team_id) == str(corner_team_id):
                        team_str = "attacking"
                    else:
                        team_str = "defending"

            # Check visibility
            is_visible = True
            if hasattr(player_data, "is_visible"):
                is_visible = bool(player_data.is_visible)

            players.append(PlayerFrame(
                player_id=player_id_str,
                team=team_str,
                role=role_str,
                x=x_m,
                y=y_m,
                is_visible=is_visible,
            ))

        # Ball coordinates
        if record.ball_coordinates is not None:
            bx, by = normalize_to_pitch(
                record.ball_coordinates.x,
                record.ball_coordinates.y,
                "skillcorner",
            )
            ball_x, ball_y = bx, by

        if not players:
            continue

        frame_obj = Frame(
            frame_idx=fid,
            timestamp_ms=(fid / FPS) * 1000.0,
            players=players,
            ball_x=ball_x,
            ball_y=ball_y,
        )
        frames.append(frame_obj)

    if len(frames) < 10:
        logger.warning(
            "Match %d corner %d: only %d frames (need >=10), skipping",
            match_id, corner_idx, len(frames),
        )
        return None

    # Sort by frame_idx
    frames.sort(key=lambda f: f.frame_idx)

    # Compute velocities
    compute_velocities_central_diff(frames, fps=FPS)

    # Determine outcome
    outcome = _determine_outcome_from_events(
        match_id, data_dir, corner_frame, corner_period,
    )

    # Find delivery frame index within our extracted frames
    delivery_idx = 0
    for i, f in enumerate(frames):
        if f.frame_idx >= corner_frame:
            delivery_idx = i
            break

    corner_id = f"skillcorner_{match_id}_corner_{corner_idx}"

    return CornerTrackingData(
        corner_id=corner_id,
        source="skillcorner",
        match_id=str(match_id),
        delivery_frame=delivery_idx,
        fps=FPS,
        outcome=outcome,
        frames=frames,
        metadata={
            "corner_team": corner_event.get("team_name", ""),
            "player_name": corner_event.get("player_name", ""),
            "period": corner_period,
            "raw_frame": corner_frame,
            "attacking_side": corner_event.get("attacking_side", ""),
        },
    )


def extract_match(
    match_id: int,
    data_dir: Path,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> List[CornerTrackingData]:
    """Extract all corners from a single SkillCorner match.

    Args:
        match_id: SkillCorner match ID
        data_dir: Path to cloned SkillCorner opendata repo
        pre_seconds: Seconds before delivery
        post_seconds: Seconds after delivery

    Returns:
        List of CornerTrackingData objects
    """
    logger.info("Loading tracking data for match %d...", match_id)
    tracking = load_match_tracking(match_id, data_dir)

    logger.info("Loading match metadata...")
    metadata = load_match_metadata(match_id, data_dir)

    logger.info("Finding corners from events CSV...")
    corners = find_corners_from_events(match_id, data_dir)
    logger.info("Found %d corners in match %d", len(corners), match_id)

    results = []
    for idx, corner_event in enumerate(corners):
        corner_data = extract_corner_window(
            tracking=tracking,
            corner_event=corner_event,
            match_metadata=metadata,
            match_id=match_id,
            corner_idx=idx,
            data_dir=data_dir,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )
        if corner_data is not None:
            results.append(corner_data)
            n_players = [len(f.players) for f in corner_data.frames]
            logger.info(
                "  Corner %d: %d frames, %.1f avg players, outcome=%s",
                idx, len(corner_data.frames),
                sum(n_players) / len(n_players) if n_players else 0,
                corner_data.outcome,
            )
        else:
            logger.warning("  Corner %d: skipped (insufficient data)", idx)

    return results


def extract_all_matches(
    data_dir: Path,
    match_ids: Optional[List[int]] = None,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> List[CornerTrackingData]:
    """Extract corners from all (or specified) SkillCorner matches.

    Args:
        data_dir: Path to cloned SkillCorner opendata repo
        match_ids: List of match IDs to process (default: all 10)
        pre_seconds: Seconds before delivery
        post_seconds: Seconds after delivery

    Returns:
        List of CornerTrackingData from all matches
    """
    if match_ids is None:
        match_ids = ALL_MATCH_IDS

    all_corners = []
    for mid in match_ids:
        try:
            corners = extract_match(
                mid, data_dir,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
            )
            all_corners.extend(corners)
            logger.info("Match %d: extracted %d corners", mid, len(corners))
        except Exception:
            logger.exception("Failed to process match %d", mid)

    logger.info("Total: %d corners from %d matches", len(all_corners), len(match_ids))
    return all_corners
