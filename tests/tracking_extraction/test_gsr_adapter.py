"""Tests for tracking_extraction/soccernet_gsr_adapter.py.

Tests GSR-specific logic: coordinate normalization, fps detection,
team resolution heuristic, delivery-at-end-of-clip, and frame parsing.
Uses synthetic data — no real .pklz or GSR outputs needed.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from tracking_extraction.core import (
    Frame,
    PlayerFrame,
    PITCH_LENGTH,
    PITCH_WIDTH,
    normalize_to_pitch,
)
from tracking_extraction.soccernet_gsr_adapter import (
    parse_gsr_output,
    _detect_corner_taker_team,
    _resolve_teams,
    _filter_position_jumps,
    gsr_to_corner_tracking,
)


# --- Helpers ---

def _make_gsr_detections(n_frames=250, players_per_frame=12, include_ball=False):
    """Create synthetic GSR detection list (COCO-style JSON format)."""
    detections = []
    for fid in range(n_frames):
        for pid in range(players_per_frame):
            team = "left" if pid < players_per_frame // 2 else "right"
            role = "goalkeeper" if pid in (0, players_per_frame // 2) else "player"
            # Cluster players near right side of pitch (typical corner)
            x = 40.0 + pid * 1.5  # center-origin: ~40 to ~58
            y = -20.0 + pid * 2.0  # center-origin: ~-20 to ~4
            detections.append({
                "image_id": fid,
                "track_id": pid,
                "attributes": {"role": role, "team": team},
                "bbox_pitch": {
                    "x_bottom_middle": x,
                    "y_bottom_middle": y,
                },
            })
        if include_ball:
            detections.append({
                "image_id": fid,
                "track_id": 999,
                "attributes": {"role": "ball", "team": "none"},
                "bbox_pitch": {
                    "x_bottom_middle": 50.0,
                    "y_bottom_middle": -30.0,
                },
            })
    return detections


def _write_gsr_json(detections, tmpdir):
    """Write detections to a temp JSON file and return its path."""
    path = Path(tmpdir) / "test_corner.json"
    with open(path, "w") as f:
        json.dump(detections, f)
    return path


def _make_frame(fid, players, ball_xy=None):
    """Create a Frame with PlayerFrame objects."""
    bx, by = ball_xy if ball_xy else (None, None)
    return Frame(
        frame_idx=fid,
        timestamp_ms=0.0,
        players=players,
        ball_x=bx,
        ball_y=by,
    )


def _make_player(pid, team, x, y, role="player"):
    return PlayerFrame(
        player_id=pid, team=team, role=role,
        x=x, y=y, vx=0.0, vy=0.0, is_visible=True,
    )


# --- Tests: GSR coordinate normalization ---

class TestGSRCoordinateNormalization:
    def test_center_origin_to_corner_origin(self):
        """GSR center-origin (0,0) maps to pitch center (52.5, 34)."""
        x, y = normalize_to_pitch(0.0, 0.0, "soccernet_gsr")
        assert x == pytest.approx(52.5)
        assert y == pytest.approx(34.0)

    def test_top_right_corner(self):
        """GSR (52.5, 34) maps to (105, 68) — top-right corner."""
        x, y = normalize_to_pitch(52.5, 34.0, "soccernet_gsr")
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(68.0)

    def test_bottom_left_corner(self):
        """GSR (-52.5, -34) maps to (0, 0) — bottom-left corner."""
        x, y = normalize_to_pitch(-52.5, -34.0, "soccernet_gsr")
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_clamping_beyond_bounds(self):
        """Values beyond pitch boundaries are clamped."""
        x, y = normalize_to_pitch(60.0, 40.0, "soccernet_gsr")
        assert x == pytest.approx(PITCH_LENGTH)  # clamped to 105
        assert y == pytest.approx(PITCH_WIDTH)  # clamped to 68

    def test_clamping_below_bounds(self):
        """Negative clamping works."""
        x, y = normalize_to_pitch(-60.0, -40.0, "soccernet_gsr")
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)


# --- Tests: parse_gsr_output ---

class TestParseGSROutput:
    def test_basic_parsing(self, tmp_path):
        """Parses detections into Frame objects with correct structure."""
        dets = _make_gsr_detections(n_frames=10, players_per_frame=12)
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        assert len(frames) == 10
        assert all(isinstance(f, Frame) for f in frames)
        # All players non-referee, non-ball → should have 12 per frame
        assert len(frames[0].players) == 12

    def test_skips_referees(self, tmp_path):
        """Referee detections are excluded from player list."""
        dets = [{
            "image_id": 0, "track_id": 1,
            "attributes": {"role": "referee", "team": "none"},
            "bbox_pitch": {"x_bottom_middle": 50.0, "y_bottom_middle": 0.0},
        }, {
            "image_id": 0, "track_id": 2,
            "attributes": {"role": "player", "team": "left"},
            "bbox_pitch": {"x_bottom_middle": 50.0, "y_bottom_middle": 0.0},
        }]
        # Duplicate player detections to get above the min-5 threshold
        for i in range(3, 8):
            dets.append({
                "image_id": 0, "track_id": i,
                "attributes": {"role": "player", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": 50.0 + i, "y_bottom_middle": 0.0},
            })
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        assert len(frames) == 1
        roles = [p.role for p in frames[0].players]
        assert "referee" not in roles

    def test_ball_becomes_frame_ball(self, tmp_path):
        """Ball detections set frame ball_x/ball_y."""
        dets = _make_gsr_detections(n_frames=5, players_per_frame=10, include_ball=True)
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        assert frames[0].ball_x is not None
        assert frames[0].ball_y is not None

    def test_min_player_filter(self, tmp_path):
        """Frames with fewer than 5 players are dropped."""
        dets = _make_gsr_detections(n_frames=5, players_per_frame=3)
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        assert len(frames) == 0  # All frames have only 3 players

    def test_timestamps_are_zero(self, tmp_path):
        """Timestamps are 0.0 (set by caller once gsr_fps is known)."""
        dets = _make_gsr_detections(n_frames=5, players_per_frame=10)
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        assert all(f.timestamp_ms == 0.0 for f in frames)

    def test_coordinates_normalized(self, tmp_path):
        """Output coordinates should be in pitch-origin (0-105, 0-68)."""
        dets = [{
            "image_id": 0, "track_id": i,
            "attributes": {"role": "player", "team": "left"},
            "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
        } for i in range(6)]
        path = _write_gsr_json(dets, tmp_path)
        frames = parse_gsr_output(path)
        # (0,0) center-origin -> (52.5, 34) pitch-origin
        assert frames[0].players[0].x == pytest.approx(52.5)
        assert frames[0].players[0].y == pytest.approx(34.0)


# --- Tests: _detect_corner_taker_team ---

class TestDetectCornerTakerTeam:
    def test_player_at_corner_flag(self):
        """Player standing on corner flag is identified as corner taker."""
        frames = [_make_frame(0, [
            _make_player("p1", "left", 0.5, 0.5),   # Near (0,0) flag
            _make_player("p2", "right", 50.0, 34.0),  # Center of pitch
        ])]
        result = _detect_corner_taker_team(frames)
        assert result == "left"

    def test_player_at_far_corner(self):
        """Player near (105, 68) corner is detected."""
        frames = [_make_frame(0, [
            _make_player("p1", "right", 104.0, 67.5),  # Near (105, 68)
            _make_player("p2", "left", 50.0, 34.0),
        ])]
        result = _detect_corner_taker_team(frames)
        assert result == "right"

    def test_no_player_near_flag(self):
        """Returns None if no player is within 10m of any flag."""
        frames = [_make_frame(0, [
            _make_player("p1", "left", 50.0, 34.0),   # Center
            _make_player("p2", "right", 50.0, 34.0),
        ])]
        result = _detect_corner_taker_team(frames)
        assert result is None

    def test_searches_all_frames(self):
        """Finds corner taker even if only visible in early frames."""
        frame_early = _make_frame(0, [
            _make_player("p1", "left", 1.0, 1.0),  # Near flag in frame 0
            _make_player("p2", "right", 50.0, 34.0),
        ])
        frame_late = _make_frame(100, [
            _make_player("p1", "left", 50.0, 34.0),  # Moved to center
            _make_player("p2", "right", 50.0, 34.0),
        ])
        result = _detect_corner_taker_team([frame_early, frame_late])
        assert result == "left"

    def test_empty_frames(self):
        """Returns None for empty frame list."""
        assert _detect_corner_taker_team([]) is None

    def test_threshold_boundary(self):
        """Player at exactly 10m is accepted, 10.1m is rejected."""
        # 10m from (0,0) along diagonal
        d = 10.0
        frames_in = [_make_frame(0, [
            _make_player("p1", "left", d / math.sqrt(2), d / math.sqrt(2)),
        ])]
        assert _detect_corner_taker_team(frames_in) == "left"

        d = 10.1
        frames_out = [_make_frame(0, [
            _make_player("p1", "left", d / math.sqrt(2), d / math.sqrt(2)),
        ])]
        assert _detect_corner_taker_team(frames_out) is None


# --- Tests: _resolve_teams ---

class TestResolveTeams:
    def test_basic_resolution(self):
        """Left=attacking maps left->attacking, right->defending."""
        players = [
            _make_player("p1", "left", 50, 34),
            _make_player("p2", "right", 60, 34),
        ]
        _resolve_teams(players, "left")
        assert players[0].team == "attacking"
        assert players[1].team == "defending"

    def test_right_attacking(self):
        """Right=attacking maps right->attacking, left->defending."""
        players = [
            _make_player("p1", "left", 50, 34),
            _make_player("p2", "right", 60, 34),
        ]
        _resolve_teams(players, "right")
        assert players[0].team == "defending"
        assert players[1].team == "attacking"

    def test_unknown_team_preserved(self):
        """Players with non-left/right teams become unknown."""
        players = [
            _make_player("p1", "nan", 50, 34),
        ]
        _resolve_teams(players, "left")
        assert players[0].team == "unknown"


# --- Tests: gsr_to_corner_tracking (integration) ---

class TestGSRToCornerTracking:
    def _make_clip_json_and_metadata(self, tmp_path, n_frames=250,
                                      players_per_frame=12):
        """Create a synthetic GSR JSON and matching metadata dict."""
        dets = _make_gsr_detections(n_frames=n_frames,
                                     players_per_frame=players_per_frame)
        path = _write_gsr_json(dets, tmp_path)
        metadata = {
            "corner_id": "corner_0000",
            "corner_idx": 0,
            "clip_path": "/fake/path.mp4",
            "corner_time_ms": 182775,
            "clip_start_ms": 152775,
            "outcome": "SHOT_ON_TARGET",
            "match_dir": "test_league/test_match",
        }
        return path, metadata

    def test_produces_corner_data(self, tmp_path):
        """Full pipeline produces a CornerTrackingData object."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        result = gsr_to_corner_tracking(path, meta)
        assert result is not None
        assert result.source == "soccernet_gsr"
        assert result.corner_id == "gsr_corner_0000"

    def test_delivery_at_end(self, tmp_path):
        """Delivery frame index is the last frame in the window."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        result = gsr_to_corner_tracking(path, meta)
        assert result.delivery_frame == len(result.frames) - 1

    def test_fps_detected_from_data(self, tmp_path):
        """FPS is auto-detected from frame count / clip duration."""
        path, meta = self._make_clip_json_and_metadata(tmp_path, n_frames=250)
        result = gsr_to_corner_tracking(path, meta, clip_duration_s=30.0)
        assert result.fps == pytest.approx(250.0 / 30.0, abs=0.1)

    def test_fps_different_frame_count(self, tmp_path):
        """FPS adapts to different frame counts."""
        path, meta = self._make_clip_json_and_metadata(tmp_path, n_frames=150)
        result = gsr_to_corner_tracking(path, meta, clip_duration_s=30.0)
        assert result.fps == pytest.approx(150.0 / 30.0, abs=0.1)

    def test_timestamps_computed(self, tmp_path):
        """Timestamps are set based on detected fps."""
        path, meta = self._make_clip_json_and_metadata(tmp_path, n_frames=250)
        result = gsr_to_corner_tracking(path, meta, clip_duration_s=30.0)
        # Last frame should be near 30s
        last_ts = result.frames[-1].timestamp_ms
        assert last_ts == pytest.approx(29880.0, rel=0.05)

    def test_shot_outcome_mapping(self, tmp_path):
        """SHOT_ON_TARGET maps to 'shot'."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        meta["outcome"] = "SHOT_ON_TARGET"
        result = gsr_to_corner_tracking(path, meta)
        assert result.outcome == "shot"

    def test_no_shot_outcome_mapping(self, tmp_path):
        """NOT_DANGEROUS maps to 'no_shot'."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        meta["outcome"] = "NOT_DANGEROUS"
        result = gsr_to_corner_tracking(path, meta)
        assert result.outcome == "no_shot"

    def test_velocities_computed(self, tmp_path):
        """Velocities are populated on player frames."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        result = gsr_to_corner_tracking(path, meta)
        # At least some players should have non-zero velocity
        # (synthetic data has constant positions so velocities will be 0,
        # but the fields should be populated)
        some_player = result.frames[1].players[0]
        assert some_player.vx is not None
        assert some_player.vy is not None

    def test_rejects_too_few_frames(self, tmp_path):
        """Returns None if not enough frames pass quality filter."""
        # Only 5 frames → way below min_frames=20
        path, meta = self._make_clip_json_and_metadata(
            tmp_path, n_frames=5, players_per_frame=12,
        )
        result = gsr_to_corner_tracking(path, meta)
        assert result is None

    def test_window_uses_pre_seconds(self, tmp_path):
        """Only frames within pre_seconds before delivery are included."""
        path, meta = self._make_clip_json_and_metadata(tmp_path, n_frames=250)
        # 5 seconds at ~8.33fps = ~42 frames
        result = gsr_to_corner_tracking(path, meta, pre_seconds=5.0)
        assert len(result.frames) <= 50  # roughly 42 + some margin

    def test_metadata_includes_gsr_fps(self, tmp_path):
        """Output metadata includes detected gsr_fps."""
        path, meta = self._make_clip_json_and_metadata(tmp_path)
        result = gsr_to_corner_tracking(path, meta)
        assert "gsr_fps" in result.metadata
        assert result.metadata["gsr_fps"] > 0


# --- Tests: _filter_position_jumps ---

class TestFilterPositionJumps:
    def test_no_jumps_passes_all(self):
        """Frames with smooth positions all pass."""
        frames = []
        for i in range(10):
            frames.append(_make_frame(i, [
                _make_player("p1", "left", 50.0 + i * 0.1, 34.0),
            ]))
        result = _filter_position_jumps(frames, max_jump_m=5.0, fps=10.0)
        assert len(result) == 10

    def test_wild_jump_filtered(self):
        """Frame with >5m jump from previous frame is filtered out."""
        frames = [
            _make_frame(0, [_make_player("p1", "left", 50.0, 34.0)]),
            _make_frame(1, [_make_player("p1", "left", 50.0, 34.0)]),
            _make_frame(2, [_make_player("p1", "left", 80.0, 34.0)]),  # 30m jump!
            _make_frame(3, [_make_player("p1", "left", 50.5, 34.0)]),
        ]
        result = _filter_position_jumps(frames, max_jump_m=5.0, fps=10.0)
        # Frame 2 should be filtered, frame 3 is relative to frame 1 (unchanged)
        assert len(result) < 4
        frame_ids = [f.frame_idx for f in result]
        assert 2 not in frame_ids
