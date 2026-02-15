"""Extract corner kick records from DFL Bundesliga tracking data.

Produces flat records in the same format as extract_corners.py (SkillCorner)
so they can be merged into a combined dataset.

Data source: 7 DFL Bundesliga matches with 25Hz optical tracking (via kloppy)
and event XML files with <CornerKick> annotations.

Each record has 22 players (no ball node), center-origin meter coordinates,
normalized so the attacking team always attacks toward +x.

Usage:
    python -m corner_prediction.data.extract_dfl_corners
    python -m corner_prediction.data.extract_dfl_corners --match-ids DFL-MAT-J03WMX
    python -m corner_prediction.data.extract_dfl_corners --output-dir corner_prediction/data
"""

import argparse
import bisect
import json
import logging
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DFL_MATCH_IDS = [
    "DFL-MAT-J03WMX",
    "DFL-MAT-J03WN1",
    "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY",
    "DFL-MAT-J03WQQ",
    "DFL-MAT-J03WR9",
]

DFL_FPS = 25.0

DFL_PITCH_LENGTH = 105.0  # meters
DFL_PITCH_WIDTH = 68.0    # meters
HALF_LENGTH = 52.5
HALF_WIDTH = 34.0

# Map kloppy starting_position names to role abbreviations
DFL_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Left Center Back": "CB",
    "Right Center Back": "CB",
    "Center Back": "CB",
    "Left Back": "LB",
    "Right Back": "RB",
    "Left Wing Back": "LWB",
    "Right Wing Back": "RWB",
    "Left Defensive Midfield": "DM",
    "Right Defensive Midfield": "DM",
    "Center Defensive Midfield": "DM",
    "Central Midfield": "DM",
    "Left Midfield": "LM",
    "Right Midfield": "RM",
    "Center Attacking Midfield": "AM",
    "Left Winger": "LW",
    "Right Winger": "RW",
    "Left Forward": "LF",
    "Right Forward": "RF",
    "Striker": "CF",
    "Center Forward": "CF",
    "Unknown": "SUB",
}


# ---------------------------------------------------------------------------
# Step 1: Parse corner events from DFL event XML
# ---------------------------------------------------------------------------

def find_corners_from_xml(event_xml_path: Path) -> List[Dict[str, Any]]:
    """Parse DFL event XML to find all corner kick events.

    Returns list of dicts with: start_frame, end_frame, team_id, side, placing.
    """
    tree = ET.parse(str(event_xml_path))
    root = tree.getroot()

    corners = []
    for event_elem in root.iter("Event"):
        ck = event_elem.find("CornerKick")
        if ck is None:
            continue

        start_frame = int(event_elem.attrib.get("StartFrame", 0))
        end_frame = int(event_elem.attrib.get("EndFrame", 0))
        team_id = ck.attrib.get("Team", "")
        side = ck.attrib.get("Side", "unknown")
        placing = ck.attrib.get("Placing", "")

        corners.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "team_id": team_id,
            "side": side,
            "placing": placing,
        })

    corners.sort(key=lambda c: c["start_frame"])
    return corners


# ---------------------------------------------------------------------------
# Step 2: Build corner → shot/goal outcome mapping from event XML
# ---------------------------------------------------------------------------

def build_corner_outcomes(event_xml_path: Path) -> Dict[int, Dict[str, bool]]:
    """Build mapping from corner start_frame to shot/goal outcome.

    DFL ShotAtGoal events have StartFrame=0 and use BallPossessionPhase
    instead of frame numbers. We match shots to corners via:
    1. ShotAtGoal events with BuildUp=cornerKick or TakerSetup=cornerKick
    2. Sequential XML ordering: each corner-shot is assigned to the most
       recent corner by the same team.

    Returns {corner_start_frame: {lead_to_shot: bool, lead_to_goal: bool}}.
    """
    tree = ET.parse(str(event_xml_path))
    root = tree.getroot()

    # Track the last corner per team as we scan events in XML order
    last_corner_frame = {}  # team_id -> most recent corner start_frame
    outcomes = {}  # corner_start_frame -> {lead_to_shot, lead_to_goal}

    for event_elem in root.iter("Event"):
        # Track corner events
        ck = event_elem.find("CornerKick")
        if ck is not None:
            frame = int(event_elem.attrib.get("StartFrame", 0))
            team = ck.attrib.get("Team", "")
            if frame > 0:
                last_corner_frame[team] = frame
                if frame not in outcomes:
                    outcomes[frame] = {"lead_to_shot": False, "lead_to_goal": False}
            continue

        # Track shot events from corners
        sag = event_elem.find("ShotAtGoal")
        if sag is not None:
            buildup = sag.attrib.get("BuildUp", "")
            setup = sag.attrib.get("TakerSetup", "")
            team = sag.attrib.get("Team", "")
            result = sag.attrib.get("Result", "")

            is_corner_shot = ("cornerKick" in buildup or "cornerKick" in setup)
            if is_corner_shot and team in last_corner_frame:
                cf = last_corner_frame[team]
                if cf in outcomes:
                    outcomes[cf]["lead_to_shot"] = True
                    if "goal" in result.lower():
                        outcomes[cf]["lead_to_goal"] = True

        # GoalScored: only attribute to corner if we already flagged a shot
        # from that corner (via ShotAtGoal with BuildUp=cornerKick).
        # Without this guard, any open-play goal would be falsely attributed
        # to the most recent corner by the same team.
        gs = event_elem.find("GoalScored")
        if gs is not None:
            team = gs.attrib.get("Team", event_elem.attrib.get("Team", ""))
            if team in last_corner_frame:
                cf = last_corner_frame[team]
                if cf in outcomes and outcomes[cf]["lead_to_shot"]:
                    outcomes[cf]["lead_to_goal"] = True

    return outcomes


# ---------------------------------------------------------------------------
# Step 3: Parse positions XML directly (bypasses kloppy period 2 bug)
# ---------------------------------------------------------------------------

def _bisect_frame(frame_ids: List[int], target: int) -> int:
    """Binary search for the frame_id closest to target."""
    idx = bisect.bisect_left(frame_ids, target)
    if idx == 0:
        return 0
    if idx == len(frame_ids):
        return len(frame_ids) - 1
    # Check which neighbor is closer
    if abs(frame_ids[idx] - target) < abs(frame_ids[idx - 1] - target):
        return idx
    return idx - 1


def parse_positions_xml(
    pos_xml_path: Path,
) -> Dict[str, Any]:
    """Parse DFL positions XML into per-person frame lookups.

    Returns dict with:
        'persons': {
            person_id: {
                'team_id': str,
                'game_section': str ('firstHalf' or 'secondHalf'),
                'frame_ids': sorted list of int,
                'positions': {frame_id: (x_m, y_m, speed_mps)},
            }
        }
        'ball': {
            game_section: {
                'frame_ids': sorted list of int,
                'positions': {frame_id: (x_m, y_m, z_m)},
            }
        }

    Coordinates in raw XML are center-origin meters (no conversion needed).
    """
    logger.info("Parsing positions XML: %s", pos_xml_path.name)
    tree = ET.parse(str(pos_xml_path))
    root = tree.getroot()
    positions_elem = root.find("Positions")

    persons = {}
    ball = {}

    for frameset in positions_elem:
        if frameset.tag != "FrameSet":
            continue

        team_id = frameset.attrib.get("TeamId", "")
        person_id = frameset.attrib.get("PersonId", "")
        game_section = frameset.attrib.get("GameSection", "")

        if team_id == "referee":
            continue

        if team_id == "BALL":
            frame_ids = []
            ball_positions = {}
            for frame_el in frameset:
                fid = int(frame_el.attrib["N"])
                x = float(frame_el.attrib["X"])
                y = float(frame_el.attrib["Y"])
                z = float(frame_el.attrib.get("Z", "0"))
                frame_ids.append(fid)
                ball_positions[fid] = (x, y, z)
            ball[game_section] = {
                "frame_ids": frame_ids,  # already sorted by XML order
                "positions": ball_positions,
            }
            continue

        # Player FrameSet
        frame_ids = []
        player_positions = {}
        for frame_el in frameset:
            fid = int(frame_el.attrib["N"])
            x = float(frame_el.attrib["X"])
            y = float(frame_el.attrib["Y"])
            speed = float(frame_el.attrib.get("S", "0"))
            frame_ids.append(fid)
            player_positions[fid] = (x, y, speed)

        # Key: person_id + game_section (same player has separate FrameSets per half)
        key = f"{person_id}_{game_section}"
        persons[key] = {
            "person_id": person_id,
            "team_id": team_id,
            "game_section": game_section,
            "frame_ids": frame_ids,
            "positions": player_positions,
        }

    n_persons = len(set(p["person_id"] for p in persons.values()))
    logger.info("  Parsed %d person-halves (%d unique players), %d ball halves",
                len(persons), n_persons, len(ball))
    return {"persons": persons, "ball": ball}


def load_player_metadata(data_dir: Path, match_id: str) -> Dict[str, Dict[str, str]]:
    """Load player metadata (team, position) via kloppy.

    Returns {person_id: {'team_id': str, 'role': str, 'name': str}}.
    """
    from kloppy import sportec

    meta_files = sorted(data_dir.glob(f"*matchinformation*{match_id}*"))
    pos_files = sorted(data_dir.glob(f"*positions_raw*{match_id}*"))

    if not meta_files or not pos_files:
        logger.warning("Missing metadata files for %s", match_id)
        return {}

    # Load kloppy just for metadata (not full tracking)
    tracking = sportec.load_tracking(
        raw_data=str(pos_files[0]),
        meta_data=str(meta_files[0]),
        limit=1,  # Load minimal frames, we only need metadata
    )

    metadata = {}
    for team in tracking.metadata.teams:
        for player in team.players:
            pid = str(player.player_id)
            pos_str = str(player.starting_position) if player.starting_position else "Unknown"
            role = DFL_POSITION_MAP.get(pos_str, "SUB")
            metadata[pid] = {
                "team_id": str(team.team_id),
                "role": role,
                "name": str(player.name) if player.name else pid,
            }

    return metadata


def get_players_at_frame(
    parsed_data: Dict,
    player_metadata: Dict[str, Dict[str, str]],
    target_frame: int,
    corner_team_id: str,
    max_gap: int = 25,  # max 1 second gap at 25fps
) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """Get all player and ball positions at a specific frame.

    Uses binary search on pre-parsed frame_ids for each person.
    Coordinates from XML are already in center-origin meters.

    Args:
        parsed_data: Output from parse_positions_xml.
        player_metadata: Output from load_player_metadata.
        target_frame: XML absolute frame number.
        corner_team_id: Team ID of the corner-taking team.
        max_gap: Maximum frame gap tolerance for nearest-frame lookup.

    Returns:
        (player_records, ball_dict) or (None, None) if insufficient data.
    """
    persons = parsed_data["persons"]
    ball_data = parsed_data["ball"]

    player_records = []
    seen_person_ids = set()

    for key, pinfo in persons.items():
        person_id = pinfo["person_id"]
        if person_id in seen_person_ids:
            continue

        frame_ids = pinfo["frame_ids"]
        if not frame_ids:
            continue

        # Binary search for nearest frame
        best_idx = _bisect_frame(frame_ids, target_frame)
        nearest_fid = frame_ids[best_idx]
        gap = abs(nearest_fid - target_frame)

        if gap > max_gap:
            continue

        x_m, y_m, speed_raw = pinfo["positions"][nearest_fid]

        # Get previous frame for velocity computation
        vx, vy = 0.0, 0.0
        if best_idx > 0:
            prev_fid = frame_ids[best_idx - 1]
            px, py, _ = pinfo["positions"][prev_fid]
            dt = (nearest_fid - prev_fid) / DFL_FPS
            if dt > 0:
                vx = (x_m - px) / dt
                vy = (y_m - py) / dt

        speed = math.sqrt(vx * vx + vy * vy)

        # Get metadata for this player
        meta = player_metadata.get(person_id, {})
        team_id = meta.get("team_id", pinfo["team_id"])
        role = meta.get("role", "SUB")
        is_gk = (role == "GK")
        is_attacking = (team_id == corner_team_id)

        player_records.append({
            "player_id": person_id,
            "x": x_m,
            "y": y_m,
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "speed": round(speed, 4),
            "is_attacking": is_attacking,
            "is_corner_taker": False,
            "is_goalkeeper": is_gk,
            "role": role,
            "is_detected": True,
            "is_receiver": False,
        })
        seen_person_ids.add(person_id)

    # Ball data
    ball = {"x": 0.0, "y": 0.0, "z": None, "is_detected": False}
    for section_data in ball_data.values():
        frame_ids = section_data["frame_ids"]
        if not frame_ids:
            continue
        best_idx = _bisect_frame(frame_ids, target_frame)
        nearest_fid = frame_ids[best_idx]
        if abs(nearest_fid - target_frame) <= max_gap:
            bx, by, bz = section_data["positions"][nearest_fid]
            ball = {"x": bx, "y": by, "z": bz, "is_detected": True}
            break

    if len(player_records) < 20:
        return None, None

    return player_records, ball


# ---------------------------------------------------------------------------
# Step 4: Direction normalization (center-origin meter coordinates)
# ---------------------------------------------------------------------------

def determine_attacking_direction(
    player_records: List[Dict],
    corner_team_id: str,
) -> str:
    """Determine which direction the corner-taking team attacks.

    Heuristic: the corner-taking team's goalkeeper should be at the
    opposite end from the corner. GK at x < 0 means team attacks +x.

    Coordinates are center-origin meters.

    Returns "left_to_right" or "right_to_left".
    """
    # Find corner-taking team's goalkeeper
    for p in player_records:
        if not p["is_attacking"]:
            continue
        if p["is_goalkeeper"]:
            # GK at negative x → team attacks toward +x
            return "left_to_right" if p["x"] < 0 else "right_to_left"

    # Fallback: average x of attacking team
    xs = [p["x"] for p in player_records if p["is_attacking"]]
    if xs:
        avg_x = sum(xs) / len(xs)
        return "left_to_right" if avg_x < 0 else "right_to_left"
    return "left_to_right"


def normalize_direction(
    players: List[Dict],
    ball: Dict,
    attacking_direction: str,
) -> Tuple[List[Dict], Dict, str]:
    """Normalize so attacking team always attacks toward +x.

    Flips all coordinates if attacking toward -x (right_to_left).
    Also determines corner_side from ball y position.

    Returns (players, ball, corner_side).
    """
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

    # Corner side from ball y (after normalization)
    # Ball should be near +x (corner flag area)
    # y > 0 = right side, y < 0 = left side
    ball_y = ball.get("y")
    if ball_y is not None:
        corner_side = "right" if ball_y > 0 else "left"
    else:
        corner_side = "unknown"

    return players, ball, corner_side


# ---------------------------------------------------------------------------
# Step 5: Extract corners for a single match
# ---------------------------------------------------------------------------

def _determine_period(target_frame: int, parsed_data: Dict) -> int:
    """Determine which period a frame belongs to based on ball data ranges."""
    for section, bdata in parsed_data["ball"].items():
        fids = bdata["frame_ids"]
        if fids and fids[0] <= target_frame <= fids[-1]:
            if section == "firstHalf":
                return 1
            elif section == "secondHalf":
                return 2
    return 1


def extract_dfl_match_corners(
    match_id: str,
    data_dir: Path,
) -> List[Dict[str, Any]]:
    """Extract all corner records from a single DFL match.

    Parses positions XML directly for tracking data (bypasses kloppy period 2 bug).
    Uses kloppy only for player metadata (team, position, name).
    """
    # Find XML files
    event_xml_files = sorted(data_dir.glob(f"*events_raw*{match_id}*"))
    pos_xml_files = sorted(data_dir.glob(f"*positions_raw*{match_id}*"))

    if not event_xml_files:
        logger.warning("No event XML for %s", match_id)
        return []
    if not pos_xml_files:
        logger.warning("No positions XML for %s", match_id)
        return []

    event_xml_path = event_xml_files[0]
    pos_xml_path = pos_xml_files[0]

    # Parse corner events from XML
    corner_events = find_corners_from_xml(event_xml_path)
    logger.info("Match %s: %d corners from XML", match_id, len(corner_events))

    if not corner_events:
        return []

    # Build corner -> outcome mapping
    corner_outcomes = build_corner_outcomes(event_xml_path)

    # Parse positions XML (direct, no kloppy for tracking)
    parsed_data = parse_positions_xml(pos_xml_path)

    # Load player metadata via kloppy (team, position)
    player_metadata = load_player_metadata(data_dir, match_id)

    records = []
    for ci, corner_ev in enumerate(corner_events):
        corner_frame = corner_ev["start_frame"]
        corner_team_id = corner_ev["team_id"]

        # Get all player positions at this frame
        player_records, ball = get_players_at_frame(
            parsed_data, player_metadata, corner_frame, corner_team_id,
        )

        if player_records is None:
            logger.warning("  Corner %d (frame=%d): insufficient player data, skipping",
                           ci, corner_frame)
            continue

        if len(player_records) < 20 or len(player_records) > 24:
            logger.warning("  Corner %d: %d players (expected ~22), skipping",
                           ci, len(player_records))
            continue

        # Trim to exactly 22 if slightly over (e.g., 23 from substitution overlap)
        if len(player_records) > 22:
            # Keep 11 per team, preferring players closer to the ball
            atk = [p for p in player_records if p["is_attacking"]]
            dfn = [p for p in player_records if not p["is_attacking"]]
            ball_x, ball_y = ball.get("x", 0), ball.get("y", 0)

            def dist_to_ball(p):
                return math.sqrt((p["x"] - ball_x) ** 2 + (p["y"] - ball_y) ** 2)

            atk.sort(key=dist_to_ball)
            dfn.sort(key=dist_to_ball)
            player_records = atk[:11] + dfn[:11]

        # Get shot/goal outcome from pre-built mapping
        outcome = corner_outcomes.get(corner_frame, {"lead_to_shot": False, "lead_to_goal": False})

        # Determine attacking direction and normalize
        atk_dir = determine_attacking_direction(
            player_records, corner_team_id,
        )
        player_records, ball, corner_side = normalize_direction(
            player_records, ball, atk_dir,
        )

        # Detection stats
        n_detected = sum(1 for p in player_records if p["is_detected"])
        n_extrapolated = len(player_records) - n_detected
        detection_rate = n_detected / len(player_records) if player_records else 0.0

        period = _determine_period(corner_frame, parsed_data)
        corner_id = f"dfl_{match_id}_{corner_frame}"

        record = {
            "match_id": match_id,
            "corner_id": corner_id,
            "period": period,
            "delivery_frame": corner_frame,
            "corner_team_id": corner_team_id,
            "corner_taker_id": None,
            "corner_side": corner_side,

            "players": player_records,

            "ball_x": ball["x"],
            "ball_y": ball["y"],
            "ball_z": ball.get("z"),
            "ball_detected": ball["is_detected"],

            "receiver_id": None,
            "has_receiver_label": False,
            "lead_to_shot": outcome["lead_to_shot"],
            "lead_to_goal": outcome["lead_to_goal"],

            "detection_rate": round(detection_rate, 4),
            "n_detected": n_detected,
            "n_extrapolated": n_extrapolated,

            "n_passing_options": 0,
            "passing_option_ids": [],
            "n_off_ball_runs": 0,
            "pass_outcome": None,

            "source": "dfl",
        }
        records.append(record)

    logger.info("Match %s: %d corners extracted", match_id, len(records))
    return records


# ---------------------------------------------------------------------------
# Step 7: Extract all matches
# ---------------------------------------------------------------------------

def extract_all_dfl_corners(
    data_dir,
    match_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract corner records from all DFL matches."""
    data_dir = Path(data_dir)
    if match_ids is None:
        match_ids = DFL_MATCH_IDS

    all_records = []
    for mid in match_ids:
        try:
            records = extract_dfl_match_corners(mid, data_dir)
            all_records.extend(records)
        except Exception:
            logger.exception("Failed to process match %s", mid)

    logger.info("Total: %d DFL corners from %d matches", len(all_records), len(match_ids))
    return all_records


# ---------------------------------------------------------------------------
# Step 8: Validation
# ---------------------------------------------------------------------------

def validate_records(records: List[Dict[str, Any]]) -> List[str]:
    """Validate DFL corner records. Returns list of warning messages."""
    warnings = []
    for r in records:
        cid = r["corner_id"]
        n = len(r["players"])

        if n != 22:
            warnings.append(f"{cid}: {n} players (expected 22)")

        for p in r["players"]:
            if abs(p["x"]) > HALF_LENGTH + 5.0:
                warnings.append(f"{cid}: player {p['player_id']} x={p['x']:.1f} out of bounds")
            if abs(p["y"]) > HALF_WIDTH + 5.0:
                warnings.append(f"{cid}: player {p['player_id']} y={p['y']:.1f} out of bounds")
            if p["speed"] > 15.0:
                warnings.append(f"{cid}: player {p['player_id']} speed={p['speed']:.1f} m/s (>15)")

        # Check team assignment quality
        n_atk = sum(1 for p in r["players"] if p["is_attacking"])
        n_def = sum(1 for p in r["players"] if not p["is_attacking"])
        if n_atk < 8 or n_atk > 14:
            warnings.append(f"{cid}: unusual team split: {n_atk} atk, {n_def} def")

    return warnings


# ---------------------------------------------------------------------------
# Step 9: Summary
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    n = len(records)
    if n == 0:
        print("No DFL corners extracted.")
        return

    n_shot = sum(1 for r in records if r["lead_to_shot"])
    n_goal = sum(1 for r in records if r["lead_to_goal"])
    matches = sorted(set(r["match_id"] for r in records))

    speeds = [p["speed"] for r in records for p in r["players"]]
    mean_speed = sum(speeds) / len(speeds) if speeds else 0

    print(f"\n{'='*50}")
    print("DFL Corner Extraction Summary")
    print(f"{'='*50}")
    print(f"Total corners:       {n}")
    print(f"Matches:             {len(matches)}")
    print(f"Shot rate:           {n_shot}/{n} ({100*n_shot/n:.1f}%)")
    print(f"Goal rate:           {n_goal}/{n} ({100*n_goal/n:.1f}%)")
    print(f"Receiver labels:     0/{n} (0.0%) — not available for DFL")
    print(f"Detection rate:      100.0% (optical tracking)")
    print(f"Mean player speed:   {mean_speed:.2f} m/s")

    print(f"\nPer-match breakdown:")
    for mid in matches:
        mc = [r for r in records if r["match_id"] == mid]
        ms = sum(1 for r in mc if r["lead_to_shot"])
        print(f"  {mid}: {len(mc)} corners, {ms} shots")

    # Corner side distribution
    sides = {}
    for r in records:
        s = r.get("corner_side", "unknown")
        sides[s] = sides.get(s, 0) + 1
    print(f"\nCorner side: {sides}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract corner kick records from DFL Bundesliga data",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/dfl",
        help="Path to DFL data directory with XML files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="corner_prediction/data",
        help="Output directory for extracted records",
    )
    parser.add_argument(
        "--match-ids", type=str, nargs="+", default=None,
        help="Specific DFL match IDs to process",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    records = extract_all_dfl_corners(data_dir, match_ids=args.match_ids)

    # Validate
    warnings = validate_records(records)
    if warnings:
        print(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  WARNING: {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    # Save
    pkl_path = output_dir / "dfl_extracted_corners.pkl"
    json_path = output_dir / "dfl_extracted_corners.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    logger.info("Saved pickle: %s", pkl_path)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    # Summary
    print_summary(records)


if __name__ == "__main__":
    main()
