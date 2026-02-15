"""Extract corner kick records from SkillCorner open data.

Produces one structured record per corner kick with:
- 22-player snapshot at delivery frame (positions, velocities, roles)
- Ball data (x, y, z)
- Labels: receiver_id, lead_to_shot, lead_to_goal
- Quality metrics: detection_rate, n_detected
- Event context: passing_options, off_ball_runs, pass_outcome

All coordinates are center-origin meters (x ∈ [-52.5, 52.5], y ∈ [-34, 34]),
normalized so the attacking team always attacks toward +x.

Usage:
    python -m corner_prediction.data.extract_corners --data-dir data/skillcorner
    python -m corner_prediction.data.extract_corners --data-dir data/skillcorner --output-dir corner_prediction/data
"""

import argparse
import csv
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MATCH_IDS = [
    2017461, 2015213, 2013725, 2011166, 2006229,
    1996435, 1953632, 1925299, 1899585, 1886347,
]

FPS = 10.0  # SkillCorner broadcast tracking rate
DT = 1.0 / FPS  # 0.1 seconds between frames

HALF_LENGTH = 52.5  # meters (pitch half-length)
HALF_WIDTH = 34.0   # meters (pitch half-width)

# Deduplication window: events within this many frames are the same corner
DEDUP_FRAMES = 20  # 2 seconds at 10Hz

# Post-corner windows for event scanning
RECEIVER_WINDOW_FRAMES = 30   # 3 seconds for receiver identification
OUTCOME_WINDOW_FRAMES = 100   # 10 seconds for shot/goal outcome

# Map SkillCorner role names -> short abbreviations
ROLE_ABBREVIATIONS = {
    "Goalkeeper": "GK",
    "Center Back": "CB",
    "Left Center Back": "CB",
    "Right Center Back": "CB",
    "Left Back": "LB",
    "Right Back": "RB",
    "Left Wing Back": "LWB",
    "Right Wing Back": "RWB",
    "Defensive Midfield": "DM",
    "Left Defensive Midfield": "DM",
    "Right Defensive Midfield": "DM",
    "Left Midfield": "LM",
    "Right Midfield": "RM",
    "Attacking Midfield": "AM",
    "Left Winger": "LW",
    "Right Winger": "RW",
    "Left Forward": "LF",
    "Right Forward": "RF",
    "Center Forward": "CF",
    "Substitute": "SUB",
}

# Coarser 4-category grouping for one-hot encoding
POSITION_GROUP_MAP = {
    "Goalkeeper": "GK",
    "Central Defender": "DEF",
    "Full Back": "DEF",
    "Midfield": "MID",
    "Center Forward": "FWD",
    "Wide Attacker": "FWD",
    "Other": "MID",  # Substitutes etc. default to MID
}


# ---------------------------------------------------------------------------
# Step 2: Load match metadata
# ---------------------------------------------------------------------------

def load_match_metadata(match_id: int, data_dir: Path) -> Dict[str, Any]:
    """Load and parse match.json for player/team metadata.

    Returns dict with:
        home_team_id, away_team_id, home_team_name, away_team_name,
        home_team_side (list of 2 strings per period),
        players: {player_id: {team_id, role_name, role_abbrev, position_group,
                              position_group_coarse, is_goalkeeper, number, short_name}}
    """
    meta_path = Path(data_dir) / "data" / "matches" / str(match_id) / f"{match_id}_match.json"
    with open(meta_path) as f:
        raw = json.load(f)

    home_id = raw["home_team"]["id"]
    away_id = raw["away_team"]["id"]

    players = {}
    for p in raw.get("players", []):
        pid = p["id"]
        role_info = p.get("player_role", {})
        role_name = role_info.get("name", "Substitute")
        pos_group = role_info.get("position_group", "Other")
        is_gk = role_name == "Goalkeeper"

        players[pid] = {
            "team_id": p["team_id"],
            "role_name": role_name,
            "role_abbrev": ROLE_ABBREVIATIONS.get(role_name, "SUB"),
            "position_group": pos_group,
            "position_group_coarse": POSITION_GROUP_MAP.get(pos_group, "MID"),
            "is_goalkeeper": is_gk,
            "number": p.get("number"),
            "short_name": p.get("short_name", ""),
            "last_name": p.get("last_name", ""),
            "first_name": p.get("first_name", ""),
        }

    return {
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": raw["home_team"].get("name", ""),
        "away_team_name": raw["away_team"].get("name", ""),
        "home_team_side": raw.get("home_team_side", ["left_to_right", "right_to_left"]),
        "players": players,
    }


# ---------------------------------------------------------------------------
# Step 3: Name -> player_id mapping
# ---------------------------------------------------------------------------

def build_name_to_id_map(metadata: Dict[str, Any]) -> Dict[str, int]:
    """Build a case-insensitive name -> player_id lookup.

    Uses short_name, last_name, and "first last" as keys.
    """
    name_map = {}
    for pid, info in metadata["players"].items():
        for name_field in ("short_name", "last_name"):
            name = info.get(name_field, "").strip()
            if name:
                name_map[name.lower()] = pid
        first = info.get("first_name", "").strip()
        last = info.get("last_name", "").strip()
        if first and last:
            name_map[f"{first} {last}".lower()] = pid
    return name_map


def resolve_name_to_id(name: str, name_map: Dict[str, int]) -> Optional[int]:
    """Resolve a player name to player_id, case-insensitive."""
    if not name:
        return None
    return name_map.get(name.strip().lower())


# ---------------------------------------------------------------------------
# Step 4: Find corners from events CSV
# ---------------------------------------------------------------------------

def _load_events_csv(match_id: int, data_dir: Path) -> List[Dict[str, str]]:
    """Load dynamic_events.csv rows into a list of dicts."""
    csv_path = Path(data_dir) / "data" / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def find_corners(
    match_id: int,
    data_dir: Path,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Identify corner kick deliveries from dynamic_events.csv.

    Returns list of dicts with: frame, period, corner_team_id, corner_team_name,
    taker_name, taker_id, gi_before.
    """
    rows = _load_events_csv(match_id, data_dir)
    home_id = metadata["home_team_id"]
    away_id = metadata["away_team_id"]
    home_name = metadata["home_team_name"]
    away_name = metadata["away_team_name"]

    raw_corners = []
    for row in rows:
        gi = row.get("game_interruption_before", "")
        if "corner" not in gi.lower():
            continue

        row_team_id = int(row["team_id"]) if row.get("team_id") else None

        if "corner_for" in gi:
            corner_team_id = row_team_id
            corner_team_name = row.get("team_shortname", "")
        else:
            # corner_against: the other team takes the corner
            if row_team_id == home_id:
                corner_team_id = away_id
                corner_team_name = away_name
            elif row_team_id == away_id:
                corner_team_id = home_id
                corner_team_name = home_name
            else:
                corner_team_id = None
                corner_team_name = "unknown"

        taker_id = int(row["player_id"]) if row.get("player_id") and "corner_for" in gi else None
        taker_name = row.get("player_name", "") if "corner_for" in gi else ""

        raw_corners.append({
            "frame": int(row["frame_start"]),
            "period": int(row["period"]),
            "corner_team_id": corner_team_id,
            "corner_team_name": corner_team_name,
            "taker_name": taker_name,
            "taker_id": taker_id,
            "gi_before": gi,
        })

    # Deduplicate: keep first event per (period, frame within 2s window).
    # Merge taker info from all events in the cluster (corner_for events
    # carry the taker's player_id).
    deduped = []
    for c in raw_corners:
        is_dup = False
        for existing in deduped:
            if (existing["period"] == c["period"]
                    and abs(existing["frame"] - c["frame"]) <= DEDUP_FRAMES):
                is_dup = True
                # Merge taker info from corner_for events
                if c["taker_id"] is not None and existing["taker_id"] is None:
                    existing["taker_id"] = c["taker_id"]
                    existing["taker_name"] = c["taker_name"]
                break
        if not is_dup:
            deduped.append(c)

    logger.info("Match %d: %d corner events -> %d after dedup",
                match_id, len(raw_corners), len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Step 5: Extract event context (receiver, outcomes, passing options)
# ---------------------------------------------------------------------------

def extract_event_context(
    events: List[Dict[str, str]],
    corner_frame: int,
    corner_period: int,
    corner_team_id: Optional[int],
    name_map: Dict[str, int],
) -> Dict[str, Any]:
    """Extract receiver, outcomes, and event context from post-corner events.

    Returns dict with: receiver_id, receiver_name, has_receiver_label,
    lead_to_shot, lead_to_goal, pass_outcome, n_passing_options,
    passing_option_ids, n_off_ball_runs.
    """
    receiver_id = None
    receiver_name = None
    lead_to_shot = False
    lead_to_goal = False
    pass_outcome = None
    n_passing_options = 0
    passing_option_ids = []
    n_off_ball_runs = 0

    # Phase 1: Find delivery event and immediate context
    for row in events:
        if int(row["period"]) != corner_period:
            continue
        frame = int(row["frame_start"])
        if frame < corner_frame:
            continue
        if frame > corner_frame + OUTCOME_WINDOW_FRAMES:
            break

        gi = row.get("game_interruption_before", "")
        event_type = row.get("event_type", "")

        # Delivery event cluster: any event at the corner frame with corner
        # annotation (both corner_for and corner_against carry targeted names)
        if frame <= corner_frame + DEDUP_FRAMES and "corner" in gi.lower():
            # Primary receiver: player_targeted_name
            targeted_name = row.get("player_targeted_name", "").strip()
            if targeted_name and receiver_id is None:
                receiver_id = resolve_name_to_id(targeted_name, name_map)
                receiver_name = targeted_name

            # Pass outcome from delivery (prefer corner_for events)
            po = row.get("pass_outcome", "").strip()
            if po and pass_outcome is None:
                pass_outcome = po

            # Passing options count from delivery event
            npo = row.get("n_passing_options", "").strip()
            if npo:
                try:
                    n_passing_options = max(n_passing_options, int(npo))
                except ValueError:
                    pass

            # Off-ball runs from delivery event
            nobr = row.get("n_off_ball_runs", "").strip()
            if nobr:
                try:
                    n_off_ball_runs = max(n_off_ball_runs, int(nobr))
                except ValueError:
                    pass

        # Secondary receiver: passing_option with targeted=True, received=True
        if (frame <= corner_frame + RECEIVER_WINDOW_FRAMES
                and event_type == "passing_option"):
            po_player_id = row.get("player_id", "").strip()
            if po_player_id:
                try:
                    passing_option_ids.append(int(po_player_id))
                except ValueError:
                    pass

            targeted = row.get("targeted", "").strip().lower() == "true"
            received = row.get("received", "").strip().lower() == "true"
            if targeted and received and receiver_id is None:
                if po_player_id:
                    try:
                        receiver_id = int(po_player_id)
                        receiver_name = row.get("player_name", "").strip()
                    except ValueError:
                        pass

        # Tertiary receiver: next player_possession after delivery
        if (frame > corner_frame
                and frame <= corner_frame + RECEIVER_WINDOW_FRAMES
                and event_type == "player_possession"
                and "corner" not in row.get("game_interruption_before", "").lower()
                and receiver_id is None):
            po_player_id = row.get("player_id", "").strip()
            if po_player_id:
                try:
                    candidate_id = int(po_player_id)
                    # Only accept if on attacking team
                    if corner_team_id is not None:
                        row_team = row.get("team_id", "").strip()
                        if row_team and int(row_team) == corner_team_id:
                            receiver_id = candidate_id
                            receiver_name = row.get("player_name", "").strip()
                except ValueError:
                    pass

        # Outcomes: scan full window
        if row.get("lead_to_shot", "").strip().lower() == "true":
            lead_to_shot = True
        if row.get("lead_to_goal", "").strip().lower() == "true":
            lead_to_goal = True

    # If lead_to_goal is True, lead_to_shot must also be True
    if lead_to_goal:
        lead_to_shot = True

    has_receiver_label = receiver_id is not None

    return {
        "receiver_id": receiver_id,
        "receiver_name": receiver_name,
        "has_receiver_label": has_receiver_label,
        "lead_to_shot": lead_to_shot,
        "lead_to_goal": lead_to_goal,
        "pass_outcome": pass_outcome,
        "n_passing_options": n_passing_options,
        "passing_option_ids": passing_option_ids,
        "n_off_ball_runs": n_off_ball_runs,
    }


# ---------------------------------------------------------------------------
# Step 6: Extract tracking snapshot at delivery
# ---------------------------------------------------------------------------

def extract_tracking_snapshot(
    match_id: int,
    data_dir: Path,
    delivery_frame: int,
    period: int,
) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """Read JSONL tracking data and extract delivery frame + previous frame.

    Returns (players_list, ball_data) or (None, None) if frame not found.
    players_list: [{player_id, x, y, is_detected, vx, vy}]
    ball_data: {x, y, z, is_detected}
    """
    jsonl_path = (
        Path(data_dir) / "data" / "matches" / str(match_id)
        / f"{match_id}_tracking_extrapolated.jsonl"
    )

    # We need delivery_frame and delivery_frame - 1 for velocity
    target_frames = {delivery_frame - 1, delivery_frame}
    found_frames = {}

    with open(jsonl_path) as f:
        for line in f:
            frame_data = json.loads(line)
            fid = frame_data.get("frame")
            fpd = frame_data.get("period")
            if fid in target_frames and fpd == period:
                found_frames[fid] = frame_data
            # Early exit once we've passed the target range
            if fid is not None and fid > delivery_frame + 10:
                break

    if delivery_frame not in found_frames:
        logger.warning("Match %d: delivery frame %d not found in tracking",
                       match_id, delivery_frame)
        return None, None

    delivery_data = found_frames[delivery_frame]
    prev_data = found_frames.get(delivery_frame - 1)

    # Build previous-frame position lookup for velocity
    prev_positions = {}
    if prev_data:
        for p in prev_data.get("player_data", []):
            prev_positions[p["player_id"]] = (p["x"], p["y"])

    # Extract players at delivery
    players = []
    for p in delivery_data.get("player_data", []):
        pid = p["player_id"]
        x, y = p["x"], p["y"]
        is_detected = p.get("is_detected", True)

        # Velocity: backward difference from t-1
        vx, vy = 0.0, 0.0
        if pid in prev_positions:
            px, py = prev_positions[pid]
            vx = (x - px) / DT
            vy = (y - py) / DT

        players.append({
            "player_id": pid,
            "x": x,
            "y": y,
            "is_detected": is_detected,
            "vx": vx,
            "vy": vy,
        })

    # Extract ball
    ball_raw = delivery_data.get("ball_data", {})
    ball = {
        "x": ball_raw.get("x"),
        "y": ball_raw.get("y"),
        "z": ball_raw.get("z"),
        "is_detected": ball_raw.get("is_detected", False),
    }

    return players, ball


# ---------------------------------------------------------------------------
# Step 7: Direction normalization
# ---------------------------------------------------------------------------

def normalize_direction(
    players: List[Dict],
    ball: Dict,
    home_team_side: List[str],
    period: int,
    corner_team_id: Optional[int],
    home_team_id: int,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict], Dict, str]:
    """Normalize so attacking team always attacks toward +x.

    Flips all coordinates if needed. Also determines corner_side (left/right).

    Returns (players, ball, corner_side).
    """
    # Determine attacking direction for the corner-taking team
    # home_team_side[0] = period 1, [1] = period 2
    home_direction = home_team_side[period - 1] if period <= len(home_team_side) else "left_to_right"

    # "left_to_right" means home team attacks toward +x in this period
    # "right_to_left" means home team attacks toward -x in this period
    if corner_team_id == home_team_id:
        attacking_direction = home_direction
    else:
        # Away team attacks opposite direction
        attacking_direction = (
            "right_to_left" if home_direction == "left_to_right" else "left_to_right"
        )

    # If attacking toward -x (right_to_left), flip everything
    need_flip = (attacking_direction == "right_to_left")

    if need_flip:
        for p in players:
            p["x"] = -p["x"]
            p["y"] = -p["y"]
            p["vx"] = -p["vx"]
            p["vy"] = -p["vy"]
        if ball.get("x") is not None:
            ball["x"] = -ball["x"]
        if ball.get("y") is not None:
            ball["y"] = -ball["y"]

    # Determine corner side from ball y position after normalization
    # Ball should be near +x end (corner flag area)
    # y > 0 = right side of goal, y < 0 = left side of goal
    ball_y = ball.get("y")
    if ball_y is not None:
        corner_side = "right" if ball_y > 0 else "left"
    else:
        corner_side = "unknown"

    return players, ball, corner_side


# ---------------------------------------------------------------------------
# Step 8: Build corner record
# ---------------------------------------------------------------------------

def build_corner_record(
    match_id: int,
    corner_event: Dict[str, Any],
    tracking_players: List[Dict],
    ball: Dict,
    event_context: Dict[str, Any],
    metadata: Dict[str, Any],
    corner_side: str,
) -> Dict[str, Any]:
    """Assemble the final corner record dict."""
    corner_team_id = corner_event["corner_team_id"]
    taker_id = corner_event.get("taker_id")
    receiver_id = event_context["receiver_id"]

    # Build per-player records enriched with metadata
    player_records = []
    n_detected = 0
    n_extrapolated = 0

    for tp in tracking_players:
        pid = tp["player_id"]
        meta = metadata["players"].get(pid, {})
        team_id = meta.get("team_id")
        is_attacking = (team_id == corner_team_id) if (team_id and corner_team_id) else False
        is_gk = meta.get("is_goalkeeper", False)
        role = meta.get("role_abbrev", "SUB")
        is_detected = tp.get("is_detected", True)

        if is_detected:
            n_detected += 1
        else:
            n_extrapolated += 1

        vx = tp.get("vx", 0.0)
        vy = tp.get("vy", 0.0)
        speed = math.sqrt(vx * vx + vy * vy)

        player_records.append({
            "player_id": pid,
            "x": tp["x"],
            "y": tp["y"],
            "vx": vx,
            "vy": vy,
            "speed": round(speed, 4),
            "is_attacking": is_attacking,
            "is_corner_taker": (pid == taker_id),
            "is_goalkeeper": is_gk,
            "role": role,
            "is_detected": is_detected,
            "is_receiver": (pid == receiver_id) if receiver_id else False,
        })

    n_players = len(player_records)
    detection_rate = n_detected / n_players if n_players > 0 else 0.0

    corner_id = f"skillcorner_{match_id}_{corner_event['period']}_{corner_event['frame']}"

    return {
        "match_id": match_id,
        "corner_id": corner_id,
        "period": corner_event["period"],
        "delivery_frame": corner_event["frame"],
        "corner_team_id": corner_team_id,
        "corner_taker_id": taker_id,
        "corner_side": corner_side,

        "players": player_records,

        "ball_x": ball.get("x"),
        "ball_y": ball.get("y"),
        "ball_z": ball.get("z"),
        "ball_detected": ball.get("is_detected", False),

        "receiver_id": receiver_id,
        "has_receiver_label": event_context["has_receiver_label"],
        "lead_to_shot": event_context["lead_to_shot"],
        "lead_to_goal": event_context["lead_to_goal"],

        "detection_rate": round(detection_rate, 4),
        "n_detected": n_detected,
        "n_extrapolated": n_extrapolated,

        "n_passing_options": event_context["n_passing_options"],
        "passing_option_ids": event_context["passing_option_ids"],
        "n_off_ball_runs": event_context["n_off_ball_runs"],
        "pass_outcome": event_context["pass_outcome"],
    }


# ---------------------------------------------------------------------------
# Step 9: Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_match_corners(
    match_id: int,
    data_dir: Path,
    min_detection_rate: float = 0.0,
) -> List[Dict[str, Any]]:
    """Extract all corner records from a single match."""
    metadata = load_match_metadata(match_id, data_dir)
    name_map = build_name_to_id_map(metadata)
    events = _load_events_csv(match_id, data_dir)
    corners = find_corners(match_id, data_dir, metadata)

    records = []
    for corner_event in corners:
        # Extract tracking snapshot
        tracking_players, ball = extract_tracking_snapshot(
            match_id, data_dir,
            corner_event["frame"], corner_event["period"],
        )
        if tracking_players is None:
            logger.warning("Match %d frame %d: no tracking data, skipping",
                           match_id, corner_event["frame"])
            continue

        # Extract event context
        event_ctx = extract_event_context(
            events, corner_event["frame"], corner_event["period"],
            corner_event["corner_team_id"], name_map,
        )

        # Normalize direction
        tracking_players, ball, corner_side = normalize_direction(
            tracking_players, ball,
            metadata["home_team_side"], corner_event["period"],
            corner_event["corner_team_id"], metadata["home_team_id"],
            metadata,
        )

        # Build record
        record = build_corner_record(
            match_id, corner_event, tracking_players, ball,
            event_ctx, metadata, corner_side,
        )

        # Filter by detection rate
        if record["detection_rate"] < min_detection_rate:
            logger.info("  Skipping %s: detection_rate=%.1f%% < %.1f%%",
                        record["corner_id"],
                        record["detection_rate"] * 100,
                        min_detection_rate * 100)
            continue

        records.append(record)

    return records


def extract_all_corners(
    data_dir: Path,
    match_ids: Optional[List[int]] = None,
    min_detection_rate: float = 0.0,
) -> List[Dict[str, Any]]:
    """Extract corner records from all matches.

    Args:
        data_dir: Path to SkillCorner data directory
        match_ids: Specific matches to process (default: all 10)
        min_detection_rate: Minimum detection rate to include (0.0-1.0)

    Returns:
        List of corner record dicts
    """
    if match_ids is None:
        match_ids = ALL_MATCH_IDS

    all_records = []
    for mid in match_ids:
        try:
            records = extract_match_corners(mid, data_dir, min_detection_rate)
            all_records.extend(records)
            logger.info("Match %d: %d corners extracted", mid, len(records))
        except Exception:
            logger.exception("Failed to process match %d", mid)

    logger.info("Total: %d corners from %d matches", len(all_records), len(match_ids))
    return all_records


# ---------------------------------------------------------------------------
# Step 10: Validation
# ---------------------------------------------------------------------------

def validate_records(records: List[Dict[str, Any]]) -> List[str]:
    """Validate corner records. Returns list of warning messages."""
    warnings = []
    for r in records:
        cid = r["corner_id"]
        n = len(r["players"])

        # Check 22 players
        if n != 22:
            warnings.append(f"{cid}: {n} players (expected 22)")

        # Check coordinates within pitch bounds
        for p in r["players"]:
            if abs(p["x"]) > HALF_LENGTH + 1.0:
                warnings.append(f"{cid}: player {p['player_id']} x={p['x']:.1f} out of bounds")
            if abs(p["y"]) > HALF_WIDTH + 1.0:
                warnings.append(f"{cid}: player {p['player_id']} y={p['y']:.1f} out of bounds")

        # Check velocities physically plausible
        for p in r["players"]:
            if p["speed"] > 15.0:
                warnings.append(f"{cid}: player {p['player_id']} speed={p['speed']:.1f} m/s (>15)")

        # Check ball position
        if r["ball_x"] is not None and abs(r["ball_x"]) > HALF_LENGTH + 5.0:
            warnings.append(f"{cid}: ball_x={r['ball_x']:.1f} far out of bounds")

        # Check exactly one receiver flagged (if has_receiver_label)
        if r["has_receiver_label"]:
            n_receivers = sum(1 for p in r["players"] if p["is_receiver"])
            if n_receivers != 1:
                warnings.append(f"{cid}: {n_receivers} players flagged as receiver (expected 1)")

        # Check at most one corner taker
        n_takers = sum(1 for p in r["players"] if p["is_corner_taker"])
        if n_takers > 1:
            warnings.append(f"{cid}: {n_takers} corner takers (expected 0-1)")

    return warnings


# ---------------------------------------------------------------------------
# Step 11: CLI
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    n = len(records)
    if n == 0:
        print("No corners extracted.")
        return

    n_shot = sum(1 for r in records if r["lead_to_shot"])
    n_goal = sum(1 for r in records if r["lead_to_goal"])
    n_receiver = sum(1 for r in records if r["has_receiver_label"])
    mean_det = sum(r["detection_rate"] for r in records) / n
    n_ball = sum(1 for r in records if r["ball_detected"])

    matches = set(r["match_id"] for r in records)

    print(f"\n{'='*50}")
    print(f"Corner Extraction Summary")
    print(f"{'='*50}")
    print(f"Total corners:       {n}")
    print(f"Matches:             {len(matches)}")
    print(f"Shot rate:           {n_shot}/{n} ({100*n_shot/n:.1f}%)")
    print(f"Goal rate:           {n_goal}/{n} ({100*n_goal/n:.1f}%)")
    print(f"Receiver labels:     {n_receiver}/{n} ({100*n_receiver/n:.1f}%)")
    print(f"Mean detection rate: {100*mean_det:.1f}%")
    print(f"Ball detected:       {n_ball}/{n} ({100*n_ball/n:.1f}%)")

    # Per-match breakdown
    print(f"\nPer-match breakdown:")
    for mid in sorted(matches):
        mc = [r for r in records if r["match_id"] == mid]
        ms = sum(1 for r in mc if r["lead_to_shot"])
        print(f"  Match {mid}: {len(mc)} corners, {ms} shots")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract corner kick records from SkillCorner data")
    parser.add_argument("--data-dir", type=str, default="data/skillcorner",
                        help="Path to SkillCorner data directory")
    parser.add_argument("--output-dir", type=str, default="corner_prediction/data",
                        help="Output directory for extracted records")
    parser.add_argument("--min-detection-rate", type=float, default=0.0,
                        help="Minimum detection rate to include (0.0-1.0)")
    parser.add_argument("--match-ids", type=int, nargs="+", default=None,
                        help="Specific match IDs to process")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    records = extract_all_corners(
        data_dir,
        match_ids=args.match_ids,
        min_detection_rate=args.min_detection_rate,
    )

    # Validate
    warnings = validate_records(records)
    if warnings:
        print(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARNING: {w}")

    # Save
    json_path = output_dir / "extracted_corners.json"
    pkl_path = output_dir / "extracted_corners.pkl"

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    logger.info("Saved pickle: %s", pkl_path)

    # Summary
    print_summary(records)


if __name__ == "__main__":
    main()
