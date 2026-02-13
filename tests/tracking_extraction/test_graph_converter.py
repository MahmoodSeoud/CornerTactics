"""Tests for tracking_extraction/graph_converter.py.

Uses synthetic CornerTrackingData so tests run without any data files.
"""

import math
import pickle

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracking_extraction.core import (
    CornerTrackingData,
    Frame,
    PlayerFrame,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from tracking_extraction.graph_converter import (
    _compute_node_features,
    _compute_edge_features,
    _detect_attack_direction,
    _normalize_angle,
    _unit_vector,
    corner_to_ussf_graph,
    convert_dataset,
    create_splits,
    save_graph_dataset,
    load_graph_dataset,
)


# --- Fixtures ---

def _make_player(pid, team, x, y, vx=0.0, vy=0.0, role="player"):
    return PlayerFrame(
        player_id=pid, team=team, role=role,
        x=x, y=y, vx=vx, vy=vy, is_visible=True,
    )


def _make_corner(
    corner_id="test_corner_0",
    match_id="test_match",
    source="test",
    n_attacking=11,
    n_defending=11,
    outcome="shot",
    attack_right=True,
):
    """Build a synthetic corner with players clustered near a goal.

    If attack_right: players near x=105 (goal at far end, no flip needed).
    If not attack_right: players near x=0 (goal at near end, flip needed).
    """
    rng = np.random.RandomState(42)
    players = []

    # Attacking team cluster
    for i in range(n_attacking):
        if attack_right:
            x = 90.0 + rng.uniform(0, 15)
            y = 20.0 + rng.uniform(0, 28)
        else:
            x = rng.uniform(0, 15)
            y = 20.0 + rng.uniform(0, 28)
        vx = rng.uniform(-2, 2)
        vy = rng.uniform(-2, 2)
        role = "goalkeeper" if i == 0 else "player"
        players.append(_make_player(f"atk_{i}", "attacking", x, y, vx, vy, role))

    # Defending team cluster (GK near own goal)
    for i in range(n_defending):
        if attack_right:
            if i == 0:  # GK near x=105
                x = 104.0
                y = 34.0
            else:
                x = 88.0 + rng.uniform(0, 17)
                y = 15.0 + rng.uniform(0, 38)
        else:
            if i == 0:  # GK near x=0
                x = 1.0
                y = 34.0
            else:
                x = rng.uniform(0, 17)
                y = 15.0 + rng.uniform(0, 38)
        vx = rng.uniform(-1, 1)
        vy = rng.uniform(-1, 1)
        role = "goalkeeper" if i == 0 else "player"
        players.append(_make_player(f"def_{i}", "defending", x, y, vx, vy, role))

    # Ball near corner flag
    if attack_right:
        ball_x, ball_y = 104.5, 0.5
    else:
        ball_x, ball_y = 0.5, 0.5

    frame = Frame(
        frame_idx=0, timestamp_ms=0.0,
        players=players,
        ball_x=ball_x, ball_y=ball_y,
    )

    return CornerTrackingData(
        corner_id=corner_id,
        source=source,
        match_id=match_id,
        delivery_frame=0,
        fps=25.0,
        frames=[frame],
        outcome=outcome,
        metadata={"corner_time_s": 100.0},
    )


def _make_dataset(n_matches=5, corners_per_match=8):
    """Build a synthetic dataset spanning multiple matches."""
    rng = np.random.RandomState(123)
    corners = []
    for m in range(n_matches):
        for c in range(corners_per_match):
            outcome = "shot" if rng.random() < 0.35 else "no_shot"
            corner = _make_corner(
                corner_id=f"m{m}_c{c}",
                match_id=f"match_{m}",
                outcome=outcome,
            )
            corners.append(corner)
    return corners


# --- Unit Vector ---

class TestUnitVector:
    def test_nonzero_velocity(self):
        ux, uy, mag = _unit_vector(3.0, 4.0)
        assert abs(mag - 5.0) < 1e-6
        assert abs(ux - 0.6) < 1e-6
        assert abs(uy - 0.8) < 1e-6

    def test_zero_velocity_returns_default(self):
        ux, uy, mag = _unit_vector(0.0, 0.0)
        assert ux == 1.0
        assert uy == 0.0
        assert mag == 0.0

    def test_unit_vector_is_unit_length(self):
        for vx, vy in [(1, 0), (0, 1), (-3, 7), (0.01, -0.02)]:
            ux, uy, _ = _unit_vector(vx, vy)
            length = math.sqrt(ux**2 + uy**2)
            assert abs(length - 1.0) < 1e-6


# --- Normalize Angle ---

class TestNormalizeAngle:
    def test_zero_angle(self):
        assert abs(_normalize_angle(0.0) - 0.5) < 1e-6

    def test_pi_maps_to_one(self):
        assert abs(_normalize_angle(math.pi) - 1.0) < 1e-6

    def test_neg_pi_maps_to_zero(self):
        assert abs(_normalize_angle(-math.pi) - 0.0) < 1e-6

    def test_range_is_zero_one(self):
        for angle in np.linspace(-math.pi, math.pi, 100):
            val = _normalize_angle(angle)
            assert 0.0 <= val <= 1.0


# --- Node Features ---

class TestComputeNodeFeatures:
    def test_shape_is_12(self):
        feat = _compute_node_features(
            50.0, 34.0, 1.0, 0.0,
            team="attacking", is_ball=False,
            ball_x_norm=0.5, ball_y_norm=0.5,
        )
        assert feat.shape == (12,)
        assert feat.dtype == np.float32

    def test_positions_normalized(self):
        feat = _compute_node_features(
            105.0, 68.0, 0.0, 0.0,
            team="attacking", is_ball=False,
            ball_x_norm=0.0, ball_y_norm=0.0,
        )
        assert abs(feat[0] - 1.0) < 1e-6  # x
        assert abs(feat[1] - 1.0) < 1e-6  # y

    def test_attacking_flag_values(self):
        for team, expected in [("attacking", 1.0), ("defending", 0.0), ("unknown", 0.5)]:
            feat = _compute_node_features(
                50, 34, 0, 0, team=team, is_ball=False,
                ball_x_norm=0.5, ball_y_norm=0.5,
            )
            assert feat[10] == expected, f"team={team}: got {feat[10]}, expected {expected}"

    def test_ball_node_flag(self):
        feat = _compute_node_features(
            50, 34, 0, 0, team="ball", is_ball=True,
            ball_x_norm=50 / PITCH_LENGTH, ball_y_norm=34 / PITCH_WIDTH,
        )
        assert feat[10] == 0.0  # ball atk_flag
        assert feat[8] == 0.0   # dist_ball = 0 for ball itself

    def test_potential_receiver_always_zero(self):
        feat = _compute_node_features(
            50, 34, 1, 1, team="attacking", is_ball=False,
            ball_x_norm=0.5, ball_y_norm=0.5,
        )
        assert feat[11] == 0.0

    def test_velocity_magnitude_clamped(self):
        # Speed = 15 m/s > MAX_VELOCITY=10
        feat = _compute_node_features(
            50, 34, 15.0, 0.0, team="attacking", is_ball=False,
            ball_x_norm=0.5, ball_y_norm=0.5,
        )
        assert feat[4] == 1.0  # clamped to 1.0

    def test_features_in_range(self):
        """All features should be in expected ranges."""
        feat = _compute_node_features(
            80.0, 25.0, 2.0, -1.5, team="attacking", is_ball=False,
            ball_x_norm=0.9, ball_y_norm=0.1,
        )
        # Positions, vel_mag, angles, distances: [0, 1]
        for i in [0, 1, 4, 5, 6, 7, 8, 9]:
            assert 0.0 <= feat[i] <= 1.0, f"Feature {i} = {feat[i]} out of [0,1]"
        # Unit vectors: [-1, 1]
        for i in [2, 3]:
            assert -1.0 <= feat[i] <= 1.0, f"Feature {i} = {feat[i]} out of [-1,1]"


# --- Edge Features ---

class TestComputeEdgeFeatures:
    def test_shape_is_6(self):
        node_i = np.zeros(12, dtype=np.float32)
        node_j = np.ones(12, dtype=np.float32) * 0.5
        feat = _compute_edge_features(node_i, node_j)
        assert feat.shape == (6,)
        assert feat.dtype == np.float32

    def test_self_distance_is_zero(self):
        node = np.array([0.5, 0.5, 1, 0, 0.3, 0.5, 0.1, 0.5, 0.1, 0.5, 1.0, 0.0],
                        dtype=np.float32)
        feat = _compute_edge_features(node, node)
        assert feat[0] == 0.0  # distance
        assert feat[1] == 0.0  # speed_diff

    def test_speed_difference_signed(self):
        node_slow = np.array([0.5, 0.5, 1, 0, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 1.0, 0.0],
                             dtype=np.float32)
        node_fast = np.array([0.6, 0.5, 1, 0, 0.8, 0.5, 0.1, 0.5, 0.1, 0.5, 0.0, 0.0],
                             dtype=np.float32)
        feat = _compute_edge_features(node_slow, node_fast)
        assert feat[1] > 0  # fast - slow > 0
        feat_rev = _compute_edge_features(node_fast, node_slow)
        assert feat_rev[1] < 0  # slow - fast < 0

    def test_distance_in_range(self):
        node_i = np.zeros(12, dtype=np.float32)  # corner (0,0)
        node_j = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        feat = _compute_edge_features(node_i, node_j)
        assert 0.0 <= feat[0] <= 1.0

    def test_angle_features_in_range(self):
        rng = np.random.RandomState(99)
        for _ in range(20):
            node_i = rng.rand(12).astype(np.float32)
            node_j = rng.rand(12).astype(np.float32)
            node_i[2:4] = node_i[2:4] * 2 - 1  # unit vector range
            node_j[2:4] = node_j[2:4] * 2 - 1
            feat = _compute_edge_features(node_i, node_j)
            for fi in [2, 3, 4, 5]:
                assert 0.0 <= feat[fi] <= 1.0, f"Edge feature {fi}={feat[fi]}"


# --- Attack Direction Detection ---

class TestDetectAttackDirection:
    def test_defending_gk_right(self):
        frame = Frame(
            frame_idx=0, timestamp_ms=0.0,
            players=[
                _make_player("gk", "defending", 104.0, 34.0, role="goalkeeper"),
                _make_player("p1", "attacking", 90.0, 30.0),
            ],
        )
        assert _detect_attack_direction(frame) == "right"

    def test_defending_gk_left(self):
        frame = Frame(
            frame_idx=0, timestamp_ms=0.0,
            players=[
                _make_player("gk", "defending", 1.0, 34.0, role="goalkeeper"),
                _make_player("p1", "attacking", 10.0, 30.0),
            ],
        )
        assert _detect_attack_direction(frame) == "left"

    def test_fallback_to_defending_mean(self):
        """When no GK, use mean defending position."""
        frame = Frame(
            frame_idx=0, timestamp_ms=0.0,
            players=[
                _make_player("d1", "defending", 80.0, 30.0),
                _make_player("d2", "defending", 90.0, 40.0),
                _make_player("a1", "attacking", 85.0, 35.0),
            ],
        )
        # Mean defending x = 85 > 52.5 → right
        assert _detect_attack_direction(frame) == "right"

    def test_fallback_to_right_when_all_unknown(self):
        frame = Frame(
            frame_idx=0, timestamp_ms=0.0,
            players=[
                _make_player("p1", "unknown", 50.0, 30.0),
            ],
        )
        assert _detect_attack_direction(frame) == "right"


# --- Full Graph Construction ---

class TestCornerToUSSFGraph:
    def test_returns_data_object(self):
        corner = _make_corner()
        graph = corner_to_ussf_graph(corner)
        assert isinstance(graph, Data)

    def test_node_count_is_players_plus_ball(self):
        corner = _make_corner(n_attacking=11, n_defending=11)
        graph = corner_to_ussf_graph(corner)
        assert graph.x.shape[0] == 23  # 22 players + 1 ball

    def test_node_features_shape(self):
        corner = _make_corner()
        graph = corner_to_ussf_graph(corner)
        assert graph.x.shape[1] == 12

    def test_dense_edge_count(self):
        corner = _make_corner(n_attacking=5, n_defending=5)
        graph = corner_to_ussf_graph(corner, adjacency="dense")
        n = 11  # 10 players + 1 ball
        assert graph.edge_index.shape == (2, n * (n - 1))

    def test_edge_attr_shape(self):
        corner = _make_corner(n_attacking=5, n_defending=5)
        graph = corner_to_ussf_graph(corner, adjacency="dense")
        n_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape == (n_edges, 6)

    def test_pos_attribute_exists(self):
        corner = _make_corner()
        graph = corner_to_ussf_graph(corner)
        assert hasattr(graph, "pos")
        assert graph.pos.shape == (graph.x.shape[0], 2)

    def test_no_nan_or_inf(self):
        corner = _make_corner()
        graph = corner_to_ussf_graph(corner)
        assert not torch.isnan(graph.x).any()
        assert not torch.isinf(graph.x).any()
        assert not torch.isnan(graph.edge_attr).any()
        assert not torch.isinf(graph.edge_attr).any()

    def test_node_feature_ranges(self):
        corner = _make_corner()
        graph = corner_to_ussf_graph(corner)
        x = graph.x.numpy()
        for i in range(12):
            if i in (2, 3):  # vx_unit, vy_unit
                assert x[:, i].min() >= -1.01
                assert x[:, i].max() <= 1.01
            elif i == 10:  # atk_flag
                vals = set(x[:, i].round(1).tolist())
                assert vals.issubset({-1.0, 0.0, 0.5, 1.0})
            else:
                assert x[:, i].min() >= -0.01, f"Feature {i} min={x[:, i].min()}"
                assert x[:, i].max() <= 1.01, f"Feature {i} max={x[:, i].max()}"

    def test_ball_node_is_last(self):
        corner = _make_corner(n_attacking=5, n_defending=5)
        graph = corner_to_ussf_graph(corner)
        # Ball node should have dist_ball = 0
        assert graph.x[-1, 8].item() == 0.0

    def test_direction_flip_normalizes_x(self):
        """Corners attacking left should be flipped so goal is at x=1.0."""
        corner_right = _make_corner(attack_right=True)
        corner_left = _make_corner(attack_right=False)
        g_right = corner_to_ussf_graph(corner_right)
        g_left = corner_to_ussf_graph(corner_left)
        # After normalization, mean x should be high (near goal) for both
        # (excluding ball node)
        mean_x_right = g_right.x[:-1, 0].mean().item()
        mean_x_left = g_left.x[:-1, 0].mean().item()
        assert mean_x_right > 0.5, f"Right attack mean x = {mean_x_right}"
        assert mean_x_left > 0.5, f"Left attack mean x = {mean_x_left}"

    def test_empty_frames_returns_none(self):
        corner = CornerTrackingData(
            corner_id="empty", source="test", match_id="m",
            delivery_frame=0, fps=25.0, frames=[], outcome="no_shot",
        )
        assert corner_to_ussf_graph(corner) is None

    def test_empty_players_returns_none(self):
        frame = Frame(frame_idx=0, timestamp_ms=0.0, players=[])
        corner = CornerTrackingData(
            corner_id="empty_players", source="test", match_id="m",
            delivery_frame=0, fps=25.0, frames=[frame], outcome="no_shot",
        )
        assert corner_to_ussf_graph(corner) is None

    def test_null_velocities_handled(self):
        """Players with None velocities should get vx=0, vy=0."""
        player = _make_player("p1", "attacking", 90.0, 34.0)
        player.vx = None
        player.vy = None
        frame = Frame(
            frame_idx=0, timestamp_ms=0.0,
            players=[
                player,
                _make_player("gk", "defending", 104.0, 34.0, role="goalkeeper"),
            ],
            ball_x=104.5, ball_y=0.5,
        )
        corner = CornerTrackingData(
            corner_id="null_vel", source="test", match_id="m",
            delivery_frame=0, fps=25.0, frames=[frame], outcome="no_shot",
        )
        graph = corner_to_ussf_graph(corner)
        assert graph is not None
        assert not torch.isnan(graph.x).any()


# --- Dataset Conversion ---

class TestConvertDataset:
    def test_returns_list_of_dicts(self):
        corners = [_make_corner(outcome="shot"), _make_corner(corner_id="c1", outcome="no_shot")]
        dataset = convert_dataset(corners)
        assert isinstance(dataset, list)
        assert len(dataset) == 2

    def test_required_keys_present(self):
        corners = [_make_corner()]
        dataset = convert_dataset(corners)
        entry = dataset[0]
        for key in ["graphs", "labels", "match_id", "corner_time", "source", "corner_id"]:
            assert key in entry, f"Missing key: {key}"

    def test_labels_correct(self):
        shot_corner = _make_corner(outcome="shot")
        noshot_corner = _make_corner(corner_id="c1", outcome="no_shot")
        dataset = convert_dataset([shot_corner, noshot_corner])
        assert dataset[0]["labels"]["shot_binary"] == 1
        assert dataset[1]["labels"]["shot_binary"] == 0

    def test_graphs_is_list_of_data(self):
        dataset = convert_dataset([_make_corner()])
        assert isinstance(dataset[0]["graphs"], list)
        assert len(dataset[0]["graphs"]) == 1
        assert isinstance(dataset[0]["graphs"][0], Data)


# --- Save / Load ---

class TestSaveLoadGraphDataset:
    def test_roundtrip(self, tmp_path):
        corners = [_make_corner()]
        dataset = convert_dataset(corners)
        path = tmp_path / "test.pkl"
        save_graph_dataset(dataset, path)
        loaded = load_graph_dataset(path)
        assert len(loaded) == 1
        assert loaded[0]["match_id"] == dataset[0]["match_id"]
        assert loaded[0]["labels"]["shot_binary"] == dataset[0]["labels"]["shot_binary"]
        assert loaded[0]["graphs"][0].x.shape == dataset[0]["graphs"][0].x.shape


# --- Splits ---

class TestCreateSplits:
    def _make_graph_dataset(self, n_matches=6, corners_per_match=10):
        """Convert synthetic corners through the full pipeline."""
        corners = _make_dataset(n_matches=n_matches, corners_per_match=corners_per_match)
        return convert_dataset(corners)

    def test_all_three_splits_present(self):
        dataset = self._make_graph_dataset()
        splits = create_splits(dataset, seed=42)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_match_overlap(self):
        dataset = self._make_graph_dataset()
        splits = create_splits(dataset, seed=42)
        train_matches = set(e["match_id"] for e in splits["train"])
        val_matches = set(e["match_id"] for e in splits["val"])
        test_matches = set(e["match_id"] for e in splits["test"])
        assert not (train_matches & val_matches), "Train-val overlap"
        assert not (train_matches & test_matches), "Train-test overlap"
        assert not (val_matches & test_matches), "Val-test overlap"

    def test_all_entries_accounted_for(self):
        dataset = self._make_graph_dataset()
        splits = create_splits(dataset, seed=42)
        total = sum(len(s) for s in splits.values())
        assert total == len(dataset)

    def test_each_split_has_both_classes(self):
        dataset = self._make_graph_dataset(n_matches=10, corners_per_match=10)
        splits = create_splits(dataset, seed=42)
        for name, entries in splits.items():
            shots = sum(1 for e in entries if e["labels"]["shot_binary"] == 1)
            no_shots = len(entries) - shots
            assert shots > 0, f"{name} has no shots"
            assert no_shots > 0, f"{name} has no no-shots"

    def test_stratification_balances_shot_rates(self):
        """Shot rates across splits should be more balanced than pure random."""
        dataset = self._make_graph_dataset(n_matches=10, corners_per_match=10)
        splits = create_splits(dataset, seed=42)
        rates = {}
        for name, entries in splits.items():
            shots = sum(1 for e in entries if e["labels"]["shot_binary"] == 1)
            rates[name] = shots / len(entries) if entries else 0

        # All split shot rates should be within 15pp of each other
        all_rates = list(rates.values())
        spread = max(all_rates) - min(all_rates)
        assert spread < 0.15, f"Shot rate spread {spread:.2f} too large: {rates}"

    def test_deterministic_with_same_seed(self):
        dataset = self._make_graph_dataset()
        splits1 = create_splits(dataset, seed=99)
        splits2 = create_splits(dataset, seed=99)
        for name in ["train", "val", "test"]:
            ids1 = [e["corner_id"] for e in splits1[name]]
            ids2 = [e["corner_id"] for e in splits2[name]]
            assert ids1 == ids2

    def test_different_seed_gives_different_splits(self):
        dataset = self._make_graph_dataset(n_matches=15, corners_per_match=8)
        splits1 = create_splits(dataset, seed=1)
        splits2 = create_splits(dataset, seed=2)
        ids1 = set(e["corner_id"] for e in splits1["test"])
        ids2 = set(e["corner_id"] for e in splits2["test"])
        assert ids1 != ids2


# --- Coordinate Transform Fix ---

class TestDFLCoordinateNormalization:
    """Verify the core.py DFL coordinate fix works correctly."""

    def test_dfl_scales_to_meters(self):
        from tracking_extraction.core import normalize_to_pitch
        # kloppy returns [0, 1] normalized; should scale to meters
        x, y = normalize_to_pitch(0.5, 0.5, "dfl")
        assert abs(x - 52.5) < 0.01  # half pitch length
        assert abs(y - 34.0) < 0.01  # half pitch width

    def test_dfl_origin(self):
        from tracking_extraction.core import normalize_to_pitch
        x, y = normalize_to_pitch(0.0, 0.0, "dfl")
        assert x == 0.0
        assert y == 0.0

    def test_dfl_far_corner(self):
        from tracking_extraction.core import normalize_to_pitch
        x, y = normalize_to_pitch(1.0, 1.0, "dfl")
        assert abs(x - 105.0) < 0.01
        assert abs(y - 68.0) < 0.01

    def test_skillcorner_center_origin(self):
        from tracking_extraction.core import normalize_to_pitch
        # SkillCorner uses center-origin meters
        x, y = normalize_to_pitch(0.0, 0.0, "skillcorner")
        assert abs(x - 52.5) < 0.01  # center of pitch
        assert abs(y - 34.0) < 0.01
