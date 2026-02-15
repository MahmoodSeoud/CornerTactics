"""Tests for corner_prediction/data/extract_corners.py.

All tests use synthetic data — no real SkillCorner files needed.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from corner_prediction.data.extract_corners import (
    ROLE_ABBREVIATIONS,
    POSITION_GROUP_MAP,
    HALF_LENGTH,
    HALF_WIDTH,
    DT,
    load_match_metadata,
    build_name_to_id_map,
    resolve_name_to_id,
    find_corners,
    extract_event_context,
    extract_tracking_snapshot,
    normalize_direction,
    build_corner_record,
    validate_records,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _make_match_json(
    home_id=100, away_id=200,
    home_name="Home FC", away_name="Away United",
    home_side=None,
    players=None,
):
    """Build a minimal match.json dict."""
    if home_side is None:
        home_side = ["left_to_right", "right_to_left"]
    if players is None:
        players = _make_players(home_id, away_id)
    return {
        "home_team": {"id": home_id, "name": home_name},
        "away_team": {"id": away_id, "name": away_name},
        "home_team_side": home_side,
        "players": players,
        "pitch_length": 105,
        "pitch_width": 68,
    }


def _make_players(home_id=100, away_id=200, n_per_team=11):
    """Generate 22 synthetic player entries for match.json."""
    roles = [
        ("Goalkeeper", "Goalkeeper"),
        ("Right Back", "Full Back"),
        ("Right Center Back", "Central Defender"),
        ("Left Center Back", "Central Defender"),
        ("Left Back", "Full Back"),
        ("Right Defensive Midfield", "Midfield"),
        ("Left Defensive Midfield", "Midfield"),
        ("Right Winger", "Wide Attacker"),
        ("Attacking Midfield", "Midfield"),
        ("Left Winger", "Wide Attacker"),
        ("Center Forward", "Center Forward"),
    ]
    players = []
    for i in range(n_per_team):
        role_name, pos_group = roles[i]
        for team_idx, team_id in enumerate([home_id, away_id]):
            pid = team_id * 100 + i
            players.append({
                "id": pid,
                "team_id": team_id,
                "player_role": {"name": role_name, "position_group": pos_group, "acronym": ""},
                "number": i + 1,
                "short_name": f"P{pid}",
                "first_name": f"First{pid}",
                "last_name": f"Last{pid}",
                "trackable_object": pid + 50000,
                "birthday": "2000-01-01",
            })
    return players


def _write_match_files(tmpdir, match_id, match_json, events_rows, tracking_frames):
    """Write synthetic match files to tmpdir."""
    match_dir = Path(tmpdir) / "data" / "matches" / str(match_id)
    match_dir.mkdir(parents=True, exist_ok=True)

    # match.json
    with open(match_dir / f"{match_id}_match.json", "w") as f:
        json.dump(match_json, f)

    # dynamic_events.csv
    if events_rows:
        fieldnames = events_rows[0].keys()
        csv_path = match_dir / f"{match_id}_dynamic_events.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events_rows)

    # tracking JSONL
    jsonl_path = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    with open(jsonl_path, "w") as f:
        for frame in tracking_frames:
            f.write(json.dumps(frame) + "\n")


def _make_event_row(**overrides):
    """Build a synthetic event CSV row."""
    defaults = {
        "frame_start": "1000",
        "period": "1",
        "event_type": "player_possession",
        "player_id": "10001",
        "player_name": "P10001",
        "team_id": "100",
        "team_shortname": "Home FC",
        "game_interruption_before": "",
        "game_interruption_after": "",
        "player_targeted_name": "",
        "pass_outcome": "",
        "lead_to_shot": "False",
        "lead_to_goal": "False",
        "targeted": "",
        "received": "",
        "n_passing_options": "",
        "n_off_ball_runs": "",
        "attacking_side": "",
    }
    defaults.update(overrides)
    return defaults


def _make_tracking_frame(frame_id, period=1, n_players=22, ball_x=50.0, ball_y=30.0,
                          ball_z=1.0, ball_detected=True):
    """Build a synthetic tracking JSONL frame."""
    players = []
    for i in range(n_players):
        team_base = 100 if i < 11 else 200
        pid = team_base * 100 + (i % 11)
        players.append({
            "player_id": pid,
            "x": 40.0 + i * 1.0,
            "y": -5.0 + i * 0.5,
            "is_detected": i % 3 != 0,  # every 3rd player extrapolated
        })
    return {
        "frame": frame_id,
        "period": period,
        "timestamp": "00:10:00.00",
        "ball_data": {
            "x": ball_x, "y": ball_y, "z": ball_z,
            "is_detected": ball_detected,
        },
        "player_data": players,
    }


# ---------------------------------------------------------------------------
# Tests: Role and name mappings (Steps 1-3)
# ---------------------------------------------------------------------------

class TestRoleMappings:
    def test_all_known_roles_have_abbreviations(self):
        """Every role in the dataset maps to a short abbreviation."""
        known_roles = [
            "Goalkeeper", "Center Back", "Left Center Back", "Right Center Back",
            "Left Back", "Right Back", "Left Wing Back", "Right Wing Back",
            "Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield",
            "Left Midfield", "Right Midfield", "Attacking Midfield",
            "Left Winger", "Right Winger", "Left Forward", "Right Forward",
            "Center Forward", "Substitute",
        ]
        for role in known_roles:
            assert role in ROLE_ABBREVIATIONS, f"Missing abbreviation for {role}"
            assert len(ROLE_ABBREVIATIONS[role]) <= 3

    def test_position_group_map_covers_all_groups(self):
        """All SkillCorner position_group values map to 4 categories."""
        groups = ["Goalkeeper", "Central Defender", "Full Back", "Midfield",
                  "Center Forward", "Wide Attacker", "Other"]
        for g in groups:
            assert g in POSITION_GROUP_MAP
        assert set(POSITION_GROUP_MAP.values()) == {"GK", "DEF", "MID", "FWD"}


class TestNameMapping:
    def test_build_name_to_id_map(self):
        mj = _make_match_json()
        meta = {
            "players": {
                p["id"]: {
                    "short_name": p["short_name"],
                    "last_name": p["last_name"],
                    "first_name": p["first_name"],
                }
                for p in mj["players"]
            },
            "home_team_id": 100,
            "away_team_id": 200,
        }
        name_map = build_name_to_id_map(meta)
        # short_name lookup (case-insensitive)
        assert resolve_name_to_id("P10000", name_map) == 10000
        assert resolve_name_to_id("p10000", name_map) == 10000
        # full name lookup
        assert resolve_name_to_id("First10000 Last10000", name_map) == 10000

    def test_resolve_empty_name_returns_none(self):
        assert resolve_name_to_id("", {}) is None
        assert resolve_name_to_id(None, {}) is None


# ---------------------------------------------------------------------------
# Tests: Corner detection (Step 4)
# ---------------------------------------------------------------------------

class TestFindCorners:
    def test_corner_for_uses_row_team(self):
        """corner_for events use the row's team_id as corner taker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mj = _make_match_json()
            events = [
                _make_event_row(
                    frame_start="1000", period="1",
                    game_interruption_before="corner_for",
                    team_id="100", player_name="Taker", player_id="10001",
                ),
            ]
            _write_match_files(tmpdir, 999, mj, events, [])
            meta = {
                "home_team_id": 100, "away_team_id": 200,
                "home_team_name": "Home FC", "away_team_name": "Away United",
                "home_team_side": ["left_to_right", "right_to_left"],
                "players": {},
            }
            corners = find_corners(999, Path(tmpdir), meta)
            assert len(corners) == 1
            assert corners[0]["corner_team_id"] == 100
            assert corners[0]["taker_id"] == 10001

    def test_corner_against_uses_opposite_team(self):
        """corner_against events assign corner to the opposing team."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mj = _make_match_json()
            events = [
                _make_event_row(
                    frame_start="2000", period="1",
                    game_interruption_before="corner_against",
                    team_id="100",
                ),
            ]
            _write_match_files(tmpdir, 999, mj, events, [])
            meta = {
                "home_team_id": 100, "away_team_id": 200,
                "home_team_name": "Home FC", "away_team_name": "Away United",
                "home_team_side": ["left_to_right", "right_to_left"],
                "players": {},
            }
            corners = find_corners(999, Path(tmpdir), meta)
            assert corners[0]["corner_team_id"] == 200

    def test_taker_merged_from_corner_for_in_cluster(self):
        """Taker ID merged from corner_for event even if corner_against comes first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mj = _make_match_json()
            events = [
                _make_event_row(
                    frame_start="1000", period="1",
                    game_interruption_before="corner_against",
                    team_id="200", player_id="20001",
                ),
                _make_event_row(
                    frame_start="1005", period="1",
                    game_interruption_before="corner_for",
                    team_id="100", player_id="10001", player_name="Taker",
                ),
            ]
            _write_match_files(tmpdir, 999, mj, events, [])
            meta = {
                "home_team_id": 100, "away_team_id": 200,
                "home_team_name": "Home FC", "away_team_name": "Away United",
                "home_team_side": ["left_to_right", "right_to_left"],
                "players": {},
            }
            corners = find_corners(999, Path(tmpdir), meta)
            assert len(corners) == 1
            assert corners[0]["taker_id"] == 10001

    def test_deduplication_within_20_frames(self):
        """Events within 20 frames in same period merge into one corner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mj = _make_match_json()
            events = [
                _make_event_row(frame_start="1000", period="1",
                                game_interruption_before="corner_for", team_id="100"),
                _make_event_row(frame_start="1005", period="1",
                                game_interruption_before="corner_against", team_id="200"),
                # Different period — not a duplicate
                _make_event_row(frame_start="1010", period="2",
                                game_interruption_before="corner_for", team_id="100"),
                # Far enough away — separate corner
                _make_event_row(frame_start="2000", period="1",
                                game_interruption_before="corner_for", team_id="100"),
            ]
            _write_match_files(tmpdir, 999, mj, events, [])
            meta = {
                "home_team_id": 100, "away_team_id": 200,
                "home_team_name": "Home FC", "away_team_name": "Away United",
                "home_team_side": ["left_to_right", "right_to_left"],
                "players": {},
            }
            corners = find_corners(999, Path(tmpdir), meta)
            assert len(corners) == 3  # frame 1000, 1010 (diff period), 2000


# ---------------------------------------------------------------------------
# Tests: Event context extraction (Step 5)
# ---------------------------------------------------------------------------

class TestEventContext:
    def test_receiver_from_targeted_name(self):
        """Primary receiver identified from player_targeted_name."""
        name_map = {"j. smith": 42}
        events = [
            _make_event_row(
                frame_start="1000", period="1",
                event_type="player_possession",
                game_interruption_before="corner_for",
                player_targeted_name="J. Smith",
                pass_outcome="successful",
                n_passing_options="5",
                n_off_ball_runs="2",
            ),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, name_map)
        assert ctx["receiver_id"] == 42
        assert ctx["receiver_name"] == "J. Smith"
        assert ctx["has_receiver_label"] is True
        assert ctx["pass_outcome"] == "successful"
        assert ctx["n_passing_options"] == 5
        assert ctx["n_off_ball_runs"] == 2

    def test_receiver_from_corner_against_event(self):
        """Receiver identified from corner_against event's player_targeted_name."""
        name_map = {"j. doe": 99}
        events = [
            _make_event_row(
                frame_start="1000", period="1",
                event_type="on_ball_engagement",
                game_interruption_before="corner_against",
                player_targeted_name="J. Doe",
            ),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, name_map)
        assert ctx["receiver_id"] == 99
        assert ctx["has_receiver_label"] is True

    def test_receiver_fallback_to_passing_option(self):
        """Secondary receiver from passing_option targeted+received."""
        events = [
            _make_event_row(
                frame_start="1000", period="1",
                event_type="player_possession",
                game_interruption_before="corner_for",
                player_targeted_name="",  # no primary
            ),
            _make_event_row(
                frame_start="1010", period="1",
                event_type="passing_option",
                player_id="555", player_name="Receiver",
                targeted="True", received="True",
            ),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, {})
        assert ctx["receiver_id"] == 555
        assert ctx["has_receiver_label"] is True

    def test_no_receiver_when_none_found(self):
        events = [
            _make_event_row(
                frame_start="1000", period="1",
                event_type="player_possession",
                game_interruption_before="corner_for",
                player_targeted_name="",
            ),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, {})
        assert ctx["receiver_id"] is None
        assert ctx["has_receiver_label"] is False

    def test_lead_to_shot_detected(self):
        events = [
            _make_event_row(frame_start="1000", period="1",
                            game_interruption_before="corner_for",
                            event_type="player_possession"),
            _make_event_row(frame_start="1050", period="1",
                            event_type="player_possession",
                            lead_to_shot="True"),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, {})
        assert ctx["lead_to_shot"] is True
        assert ctx["lead_to_goal"] is False

    def test_lead_to_goal_implies_shot(self):
        events = [
            _make_event_row(frame_start="1000", period="1",
                            game_interruption_before="corner_for",
                            event_type="player_possession"),
            _make_event_row(frame_start="1080", period="1",
                            event_type="player_possession",
                            lead_to_goal="True"),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, {})
        assert ctx["lead_to_shot"] is True
        assert ctx["lead_to_goal"] is True

    def test_passing_option_ids_collected(self):
        events = [
            _make_event_row(frame_start="1000", period="1",
                            game_interruption_before="corner_for",
                            event_type="player_possession"),
            _make_event_row(frame_start="1005", period="1",
                            event_type="passing_option", player_id="10"),
            _make_event_row(frame_start="1010", period="1",
                            event_type="passing_option", player_id="20"),
        ]
        ctx = extract_event_context(events, 1000, 1, 100, {})
        assert ctx["passing_option_ids"] == [10, 20]


# ---------------------------------------------------------------------------
# Tests: Tracking snapshot (Step 6)
# ---------------------------------------------------------------------------

class TestTrackingSnapshot:
    def test_extracts_22_players(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [
                _make_tracking_frame(999, period=1),
                _make_tracking_frame(1000, period=1),
            ]
            match_dir = Path(tmpdir) / "data" / "matches" / "999"
            match_dir.mkdir(parents=True)
            with open(match_dir / "999_tracking_extrapolated.jsonl", "w") as f:
                for fr in frames:
                    f.write(json.dumps(fr) + "\n")

            players, ball = extract_tracking_snapshot(999, Path(tmpdir), 1000, 1)
            assert len(players) == 22
            assert ball["is_detected"] is True

    def test_velocity_backward_difference(self):
        """Velocity computed as (x[t] - x[t-1]) / dt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Two frames: player moves from x=10 to x=11 (dx=1 in 0.1s = 10 m/s)
            f0 = _make_tracking_frame(99, period=1)
            f1 = _make_tracking_frame(100, period=1)
            # Override first player position
            f0["player_data"][0]["x"] = 10.0
            f0["player_data"][0]["y"] = 5.0
            f1["player_data"][0]["x"] = 11.0
            f1["player_data"][0]["y"] = 5.5

            match_dir = Path(tmpdir) / "data" / "matches" / "999"
            match_dir.mkdir(parents=True)
            with open(match_dir / "999_tracking_extrapolated.jsonl", "w") as f:
                f.write(json.dumps(f0) + "\n")
                f.write(json.dumps(f1) + "\n")

            players, _ = extract_tracking_snapshot(999, Path(tmpdir), 100, 1)
            p0 = players[0]
            assert abs(p0["vx"] - 10.0) < 0.01
            assert abs(p0["vy"] - 5.0) < 0.01

    def test_returns_none_if_frame_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            match_dir = Path(tmpdir) / "data" / "matches" / "999"
            match_dir.mkdir(parents=True)
            with open(match_dir / "999_tracking_extrapolated.jsonl", "w") as f:
                f.write(json.dumps(_make_tracking_frame(500, period=1)) + "\n")

            players, ball = extract_tracking_snapshot(999, Path(tmpdir), 1000, 1)
            assert players is None
            assert ball is None


# ---------------------------------------------------------------------------
# Tests: Direction normalization (Step 7)
# ---------------------------------------------------------------------------

class TestDirectionNormalization:
    def _make_players_and_ball(self):
        players = [{"x": 40.0, "y": 10.0, "vx": 5.0, "vy": 2.0, "player_id": 1}]
        ball = {"x": 50.0, "y": 30.0, "z": 1.5, "is_detected": True}
        return players, ball

    def test_no_flip_when_attacking_left_to_right(self):
        """Home team attacking left_to_right in period 1, home takes corner => no flip."""
        players, ball = self._make_players_and_ball()
        meta = {"players": {}}
        p, b, side = normalize_direction(
            players, ball,
            home_team_side=["left_to_right", "right_to_left"],
            period=1, corner_team_id=100, home_team_id=100, metadata=meta,
        )
        assert p[0]["x"] == 40.0  # unchanged
        assert p[0]["vx"] == 5.0
        assert b["x"] == 50.0
        assert side == "right"  # ball_y=30 > 0

    def test_flip_when_attacking_right_to_left(self):
        """Home team attacking right_to_left in period 1, home takes corner => flip."""
        players, ball = self._make_players_and_ball()
        meta = {"players": {}}
        p, b, side = normalize_direction(
            players, ball,
            home_team_side=["right_to_left", "left_to_right"],
            period=1, corner_team_id=100, home_team_id=100, metadata=meta,
        )
        assert p[0]["x"] == -40.0  # flipped
        assert p[0]["y"] == -10.0
        assert p[0]["vx"] == -5.0
        assert p[0]["vy"] == -2.0
        assert b["x"] == -50.0
        assert b["y"] == -30.0

    def test_away_team_direction_inverted(self):
        """Away team direction is opposite of home."""
        players, ball = self._make_players_and_ball()
        meta = {"players": {}}
        # Home attacks left_to_right => away attacks right_to_left => need flip
        p, b, side = normalize_direction(
            players, ball,
            home_team_side=["left_to_right", "right_to_left"],
            period=1, corner_team_id=200, home_team_id=100, metadata=meta,
        )
        assert p[0]["x"] == -40.0  # flipped because away attacks right_to_left

    def test_corner_side_left_when_ball_y_negative(self):
        players = [{"x": 40.0, "y": -10.0, "vx": 0.0, "vy": 0.0, "player_id": 1}]
        ball = {"x": 50.0, "y": -30.0, "z": 1.0, "is_detected": True}
        meta = {"players": {}}
        _, _, side = normalize_direction(
            players, ball,
            home_team_side=["left_to_right", "right_to_left"],
            period=1, corner_team_id=100, home_team_id=100, metadata=meta,
        )
        assert side == "left"

    def test_corner_side_unknown_when_ball_missing(self):
        players = [{"x": 40.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "player_id": 1}]
        ball = {"x": None, "y": None, "z": None, "is_detected": False}
        meta = {"players": {}}
        _, _, side = normalize_direction(
            players, ball,
            home_team_side=["left_to_right", "right_to_left"],
            period=1, corner_team_id=100, home_team_id=100, metadata=meta,
        )
        assert side == "unknown"


# ---------------------------------------------------------------------------
# Tests: Build record (Step 8)
# ---------------------------------------------------------------------------

class TestBuildRecord:
    def test_record_has_22_players(self):
        mj = _make_match_json()
        meta = load_match_metadata.__wrapped__(mj) if hasattr(load_match_metadata, '__wrapped__') else None
        # Build metadata manually
        meta = {
            "home_team_id": 100,
            "away_team_id": 200,
            "home_team_name": "Home FC",
            "away_team_name": "Away United",
            "home_team_side": ["left_to_right", "right_to_left"],
            "players": {},
        }
        for p in mj["players"]:
            role_info = p.get("player_role", {})
            role_name = role_info.get("name", "Substitute")
            meta["players"][p["id"]] = {
                "team_id": p["team_id"],
                "role_name": role_name,
                "role_abbrev": ROLE_ABBREVIATIONS.get(role_name, "SUB"),
                "position_group": role_info.get("position_group", "Other"),
                "is_goalkeeper": role_name == "Goalkeeper",
                "number": p.get("number"),
                "short_name": p.get("short_name", ""),
            }

        tracking_players = []
        for p in mj["players"]:
            tracking_players.append({
                "player_id": p["id"],
                "x": 40.0, "y": 5.0,
                "vx": 1.0, "vy": 0.5,
                "is_detected": True,
            })

        corner_event = {"frame": 1000, "period": 1, "corner_team_id": 100, "taker_id": 10001}
        ball = {"x": 50.0, "y": 30.0, "z": 1.5, "is_detected": True}
        event_ctx = {
            "receiver_id": 10005, "receiver_name": "P10005",
            "has_receiver_label": True, "lead_to_shot": True,
            "lead_to_goal": False, "pass_outcome": "successful",
            "n_passing_options": 4, "passing_option_ids": [10002, 10003],
            "n_off_ball_runs": 1,
        }

        record = build_corner_record(999, corner_event, tracking_players, ball,
                                      event_ctx, meta, "right")
        assert len(record["players"]) == 22
        assert record["corner_id"] == "skillcorner_999_1_1000"
        assert record["lead_to_shot"] is True
        assert record["has_receiver_label"] is True

    def test_speed_computation(self):
        meta = {"players": {1: {
            "team_id": 100, "role_abbrev": "CF", "is_goalkeeper": False,
        }}, "home_team_id": 100, "away_team_id": 200}
        tracking_players = [{
            "player_id": 1, "x": 0.0, "y": 0.0,
            "vx": 3.0, "vy": 4.0, "is_detected": True,
        }]
        corner_event = {"frame": 100, "period": 1, "corner_team_id": 100, "taker_id": None}
        ball = {"x": 50.0, "y": 30.0, "z": 1.0, "is_detected": True}
        event_ctx = {
            "receiver_id": None, "receiver_name": None,
            "has_receiver_label": False, "lead_to_shot": False,
            "lead_to_goal": False, "pass_outcome": None,
            "n_passing_options": 0, "passing_option_ids": [],
            "n_off_ball_runs": 0,
        }
        record = build_corner_record(1, corner_event, tracking_players, ball,
                                      event_ctx, meta, "right")
        assert abs(record["players"][0]["speed"] - 5.0) < 0.01

    def test_detection_rate(self):
        meta = {"players": {}, "home_team_id": 100, "away_team_id": 200}
        tracking_players = [
            {"player_id": i, "x": 0, "y": 0, "vx": 0, "vy": 0,
             "is_detected": i < 15}
            for i in range(22)
        ]
        corner_event = {"frame": 100, "period": 1, "corner_team_id": 100, "taker_id": None}
        ball = {"x": 50.0, "y": 30.0, "z": 1.0, "is_detected": True}
        event_ctx = {
            "receiver_id": None, "receiver_name": None,
            "has_receiver_label": False, "lead_to_shot": False,
            "lead_to_goal": False, "pass_outcome": None,
            "n_passing_options": 0, "passing_option_ids": [],
            "n_off_ball_runs": 0,
        }
        record = build_corner_record(1, corner_event, tracking_players, ball,
                                      event_ctx, meta, "right")
        assert record["n_detected"] == 15
        assert record["n_extrapolated"] == 7
        assert abs(record["detection_rate"] - 15 / 22) < 0.001


# ---------------------------------------------------------------------------
# Tests: Validation (Step 10)
# ---------------------------------------------------------------------------

class TestValidation:
    def _make_valid_record(self, n_players=22):
        return {
            "corner_id": "test_1",
            "players": [
                {"player_id": i, "x": 40.0, "y": 5.0, "vx": 1.0, "vy": 0.5,
                 "speed": 1.12, "is_receiver": False, "is_corner_taker": False}
                for i in range(n_players)
            ],
            "ball_x": 50.0, "ball_y": 30.0,
            "has_receiver_label": False,
        }

    def test_valid_record_no_warnings(self):
        record = self._make_valid_record()
        warnings = validate_records([record])
        assert len(warnings) == 0

    def test_wrong_player_count(self):
        record = self._make_valid_record(n_players=20)
        warnings = validate_records([record])
        assert any("20 players" in w for w in warnings)

    def test_out_of_bounds_coordinates(self):
        record = self._make_valid_record()
        record["players"][0]["x"] = 60.0  # > 52.5 + 1.0
        warnings = validate_records([record])
        assert any("out of bounds" in w for w in warnings)

    def test_excessive_speed_flagged(self):
        record = self._make_valid_record()
        record["players"][0]["speed"] = 20.0
        warnings = validate_records([record])
        assert any("speed=20.0" in w for w in warnings)

    def test_receiver_count_mismatch(self):
        record = self._make_valid_record()
        record["has_receiver_label"] = True
        record["players"][0]["is_receiver"] = True
        record["players"][1]["is_receiver"] = True  # two receivers
        warnings = validate_records([record])
        assert any("2 players flagged as receiver" in w for w in warnings)
