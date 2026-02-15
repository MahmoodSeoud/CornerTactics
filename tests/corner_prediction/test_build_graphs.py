"""Tests for corner_prediction/data/build_graphs.py.

All tests use synthetic corner records — no real SkillCorner data needed.
"""

import math

import numpy as np
import pytest
import torch

from corner_prediction.data.build_graphs import (
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    HALF_LENGTH,
    HALF_WIDTH,
    POSITION_GROUPS,
    ROLE_TO_GROUP,
    build_node_features,
    build_edge_features,
    build_knn_edges,
    build_dense_edges,
    corner_record_to_graph,
    build_graph_dataset,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_player(
    player_id=1,
    x=0.0, y=0.0,
    vx=0.0, vy=0.0,
    speed=None,
    is_attacking=True,
    is_corner_taker=False,
    is_goalkeeper=False,
    role="CB",
    is_detected=True,
    is_receiver=False,
):
    """Build a single player dict matching extract_corners.py output."""
    if speed is None:
        speed = math.sqrt(vx * vx + vy * vy)
    return {
        "player_id": player_id,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": round(speed, 4),
        "is_attacking": is_attacking,
        "is_corner_taker": is_corner_taker,
        "is_goalkeeper": is_goalkeeper,
        "role": role,
        "is_detected": is_detected,
        "is_receiver": is_receiver,
    }


def _make_corner_record(
    match_id=12345,
    n_attacking=11,
    n_defending=11,
    has_receiver=True,
    lead_to_shot=False,
    lead_to_goal=False,
    corner_side="right",
    seed=42,
):
    """Build a full corner record with 22 players in realistic positions."""
    rng = np.random.RandomState(seed)
    players = []

    # Attacking players near the goal (positive x side)
    atk_roles = ["GK", "CB", "CB", "LB", "RB", "DM", "CM", "LW", "RW", "CF", "AM"]
    for i in range(n_attacking):
        role = atk_roles[i] if i < len(atk_roles) else "MID"
        is_gk = (role == "GK")
        # Attacking GK stays back; others cluster in box
        if is_gk:
            x = rng.uniform(-50, -40)
            y = rng.uniform(-5, 5)
        else:
            x = rng.uniform(30, 52)
            y = rng.uniform(-30, 30)
        vx = rng.uniform(-2, 2)
        vy = rng.uniform(-2, 2)
        players.append(_make_player(
            player_id=100 + i,
            x=x, y=y, vx=vx, vy=vy,
            is_attacking=True,
            is_corner_taker=(i == 1 and n_attacking > 1),
            is_goalkeeper=is_gk,
            role=role,
            is_detected=bool(rng.random() > 0.2),
            is_receiver=(has_receiver and i == 9),  # CF is receiver
        ))

    # Defending players near their goal
    def_roles = ["GK", "CB", "CB", "LB", "RB", "DM", "CM", "LW", "RW", "CF", "AM"]
    for i in range(n_defending):
        role = def_roles[i] if i < len(def_roles) else "MID"
        is_gk = (role == "GK")
        if is_gk:
            x = rng.uniform(48, 52)
            y = rng.uniform(-3, 3)
        else:
            x = rng.uniform(30, 52)
            y = rng.uniform(-30, 30)
        vx = rng.uniform(-2, 2)
        vy = rng.uniform(-2, 2)
        players.append(_make_player(
            player_id=200 + i,
            x=x, y=y, vx=vx, vy=vy,
            is_attacking=False,
            is_goalkeeper=is_gk,
            role=role,
            is_detected=bool(rng.random() > 0.2),
        ))

    receiver_id = 109 if has_receiver else None

    return {
        "match_id": match_id,
        "corner_id": f"test_{match_id}_1_100",
        "period": 1,
        "delivery_frame": 100,
        "corner_team_id": 10,
        "corner_taker_id": 101,
        "corner_side": corner_side,
        "players": players,
        "ball_x": 52.0,
        "ball_y": 33.0,
        "ball_z": 0.1,
        "ball_detected": True,
        "receiver_id": receiver_id,
        "has_receiver_label": has_receiver,
        "lead_to_shot": lead_to_shot,
        "lead_to_goal": lead_to_goal,
        "detection_rate": 0.75,
        "n_detected": 16,
        "n_extrapolated": 6,
        "n_passing_options": 3,
        "passing_option_ids": [103, 105, 109],
        "n_off_ball_runs": 2,
        "pass_outcome": "successful",
    }


# ---------------------------------------------------------------------------
# TestNodeFeatures
# ---------------------------------------------------------------------------

class TestNodeFeatures:
    """Node feature construction."""

    def test_feature_dimension(self):
        p = _make_player()
        feats = build_node_features(p)
        assert len(feats) == NODE_FEATURE_DIM

    def test_position_normalization(self):
        """x=52.5 → 1.0, x=-52.5 → -1.0, y=34 → 1.0."""
        p = _make_player(x=HALF_LENGTH, y=HALF_WIDTH)
        feats = build_node_features(p)
        assert feats[0] == pytest.approx(1.0)
        assert feats[1] == pytest.approx(1.0)

        p2 = _make_player(x=-HALF_LENGTH, y=-HALF_WIDTH)
        feats2 = build_node_features(p2)
        assert feats2[0] == pytest.approx(-1.0)
        assert feats2[1] == pytest.approx(-1.0)

    def test_center_position(self):
        p = _make_player(x=0.0, y=0.0)
        feats = build_node_features(p)
        assert feats[0] == pytest.approx(0.0)
        assert feats[1] == pytest.approx(0.0)

    def test_velocity_passthrough(self):
        """Velocities are passed through as-is."""
        p = _make_player(vx=3.5, vy=-1.2, speed=3.7)
        feats = build_node_features(p)
        assert feats[2] == pytest.approx(3.5)
        assert feats[3] == pytest.approx(-1.2)
        assert feats[4] == pytest.approx(3.7)

    def test_binary_flags(self):
        p = _make_player(
            is_attacking=True,
            is_corner_taker=True,
            is_goalkeeper=False,
            is_detected=True,
        )
        feats = build_node_features(p)
        assert feats[5] == 1.0  # is_attacking
        assert feats[6] == 1.0  # is_corner_taker
        assert feats[7] == 0.0  # is_goalkeeper
        assert feats[8] == 1.0  # is_detected

    def test_binary_flags_false(self):
        p = _make_player(
            is_attacking=False,
            is_corner_taker=False,
            is_goalkeeper=True,
            is_detected=False,
        )
        feats = build_node_features(p)
        assert feats[5] == 0.0
        assert feats[6] == 0.0
        assert feats[7] == 1.0
        assert feats[8] == 0.0

    def test_position_group_onehot(self):
        """Exactly one of 4 group features is 1.0."""
        for role, expected_group in ROLE_TO_GROUP.items():
            p = _make_player(role=role)
            feats = build_node_features(p)
            group_feats = feats[9:13]
            assert sum(group_feats) == pytest.approx(1.0), f"role={role}"
            idx = POSITION_GROUPS.index(expected_group)
            assert group_feats[idx] == 1.0, f"role={role} → {expected_group}"

    def test_unknown_role_defaults_mid(self):
        p = _make_player(role="UNKNOWN_ROLE")
        feats = build_node_features(p)
        group_feats = feats[9:13]
        mid_idx = POSITION_GROUPS.index("MID")
        assert group_feats[mid_idx] == 1.0
        assert sum(group_feats) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestEdgeConstruction
# ---------------------------------------------------------------------------

class TestEdgeConstruction:
    """Edge index construction."""

    def test_knn_edge_count(self):
        """22 nodes, k=6 → 132 directed edges."""
        positions = np.random.RandomState(42).rand(22, 2)
        edge_index = build_knn_edges(positions, k=6)
        assert edge_index.shape == (2, 22 * 6)

    def test_knn_degenerates_to_dense(self):
        """5 nodes, k=6 → fully connected = 20 edges."""
        positions = np.random.RandomState(42).rand(5, 2)
        edge_index = build_knn_edges(positions, k=6)
        assert edge_index.shape == (2, 5 * 4)  # 5*(5-1)

    def test_dense_edge_count(self):
        """22 nodes → 462 directed edges."""
        edge_index = build_dense_edges(22)
        assert edge_index.shape == (2, 22 * 21)

    def test_no_self_loops_knn(self):
        positions = np.random.RandomState(42).rand(22, 2)
        edge_index = build_knn_edges(positions, k=6)
        src, dst = edge_index[0], edge_index[1]
        assert (src == dst).sum().item() == 0

    def test_no_self_loops_dense(self):
        edge_index = build_dense_edges(22)
        src, dst = edge_index[0], edge_index[1]
        assert (src == dst).sum().item() == 0

    def test_edge_indices_in_range(self):
        n = 22
        positions = np.random.RandomState(42).rand(n, 2)
        edge_index = build_knn_edges(positions, k=6)
        assert edge_index.min().item() >= 0
        assert edge_index.max().item() < n


# ---------------------------------------------------------------------------
# TestEdgeFeatures
# ---------------------------------------------------------------------------

class TestEdgeFeatures:
    """Edge feature computation."""

    def test_dimension(self):
        feats = build_edge_features(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        assert len(feats) == EDGE_FEATURE_DIM

    def test_same_team(self):
        feats = build_edge_features(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        assert feats[3] == 1.0  # both attacking

    def test_different_team(self):
        feats = build_edge_features(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        assert feats[3] == 0.0  # different teams

    def test_distance_nonnegative(self):
        feats = build_edge_features(0.0, 0.0, 1.0, 0.5, 0.3, 1.0)
        assert feats[2] >= 0.0

    def test_distance_zero_same_position(self):
        feats = build_edge_features(0.5, 0.3, 1.0, 0.5, 0.3, 1.0)
        assert feats[2] == pytest.approx(0.0)

    def test_dx_dy_antisymmetric(self):
        """dx(i→j) = -dx(j→i), dy(i→j) = -dy(j→i)."""
        f_ij = build_edge_features(0.1, 0.2, 1.0, 0.5, 0.8, 0.0)
        f_ji = build_edge_features(0.5, 0.8, 0.0, 0.1, 0.2, 1.0)
        assert f_ij[0] == pytest.approx(-f_ji[0])  # dx
        assert f_ij[1] == pytest.approx(-f_ji[1])  # dy
        assert f_ij[2] == pytest.approx(f_ji[2])   # distance is symmetric

    def test_dx_dy_values(self):
        feats = build_edge_features(0.1, 0.2, 1.0, 0.4, 0.7, 1.0)
        assert feats[0] == pytest.approx(0.3)  # dx = 0.4 - 0.1
        assert feats[1] == pytest.approx(0.5)  # dy = 0.7 - 0.2
        expected_dist = math.sqrt(0.3**2 + 0.5**2)
        assert feats[2] == pytest.approx(expected_dist)


# ---------------------------------------------------------------------------
# TestGraphConstruction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    """Full graph construction from corner records."""

    def test_full_graph_shape_knn(self):
        record = _make_corner_record()
        g = corner_record_to_graph(record, edge_type="knn", k=6)
        assert g.x.shape == (22, NODE_FEATURE_DIM)
        assert g.edge_index.shape == (2, 22 * 6)
        assert g.edge_attr.shape == (22 * 6, EDGE_FEATURE_DIM)

    def test_full_graph_shape_dense(self):
        record = _make_corner_record()
        g = corner_record_to_graph(record, edge_type="dense")
        assert g.x.shape == (22, NODE_FEATURE_DIM)
        assert g.edge_index.shape == (2, 22 * 21)
        assert g.edge_attr.shape == (22 * 21, EDGE_FEATURE_DIM)

    def test_receiver_mask_only_attacking_outfield(self):
        record = _make_corner_record()
        g = corner_record_to_graph(record)
        for i, p in enumerate(record["players"]):
            expected = p["is_attacking"] and not p["is_goalkeeper"]
            assert g.receiver_mask[i].item() == expected, (
                f"player {i}: is_attacking={p['is_attacking']}, "
                f"is_gk={p['is_goalkeeper']}, mask={g.receiver_mask[i].item()}"
            )

    def test_receiver_label_one_hot(self):
        record = _make_corner_record(has_receiver=True)
        g = corner_record_to_graph(record)
        assert g.has_receiver_label is True
        assert g.receiver_label.sum().item() == pytest.approx(1.0)
        # Check it's on the right player
        receiver_idx = g.receiver_label.argmax().item()
        assert record["players"][receiver_idx]["is_receiver"] is True

    def test_no_receiver_label(self):
        record = _make_corner_record(has_receiver=False)
        g = corner_record_to_graph(record)
        assert g.has_receiver_label is False
        assert g.receiver_label.sum().item() == pytest.approx(0.0)

    def test_receiver_on_defender_invalidated(self):
        """When the receiver is a defender, has_receiver_label should be False."""
        record = _make_corner_record(has_receiver=False)
        # Manually mark a defending player as receiver
        def_player = next(p for p in record["players"] if not p["is_attacking"])
        def_player["is_receiver"] = True
        record["has_receiver_label"] = True
        record["receiver_id"] = def_player["player_id"]

        g = corner_record_to_graph(record)
        # The receiver is outside the mask, so should be invalidated
        assert g.has_receiver_label is False
        assert g.receiver_label.sum().item() == pytest.approx(0.0)

    def test_receiver_on_goalkeeper_invalidated(self):
        """When the receiver is the attacking GK, has_receiver_label should be False."""
        record = _make_corner_record(has_receiver=False)
        # Mark the attacking GK as receiver
        atk_gk = next(
            p for p in record["players"]
            if p["is_attacking"] and p["is_goalkeeper"]
        )
        atk_gk["is_receiver"] = True
        record["has_receiver_label"] = True
        record["receiver_id"] = atk_gk["player_id"]

        g = corner_record_to_graph(record)
        assert g.has_receiver_label is False
        assert g.receiver_label.sum().item() == pytest.approx(0.0)

    def test_shot_label(self):
        record_shot = _make_corner_record(lead_to_shot=True, lead_to_goal=True)
        g_shot = corner_record_to_graph(record_shot)
        assert g_shot.shot_label == 1
        assert g_shot.goal_label == 1

        record_no = _make_corner_record(lead_to_shot=False, lead_to_goal=False)
        g_no = corner_record_to_graph(record_no)
        assert g_no.shot_label == 0
        assert g_no.goal_label == 0

    def test_corner_side(self):
        record_r = _make_corner_record(corner_side="right")
        g_r = corner_record_to_graph(record_r)
        assert g_r.corner_side == 1.0

        record_l = _make_corner_record(corner_side="left")
        g_l = corner_record_to_graph(record_l)
        assert g_l.corner_side == 0.0

    def test_metadata_preserved(self):
        record = _make_corner_record(match_id=99999)
        g = corner_record_to_graph(record)
        assert g.match_id == "99999"
        assert g.corner_id == record["corner_id"]
        assert g.detection_rate == record["detection_rate"]

    def test_no_nan_inf(self):
        record = _make_corner_record()
        g = corner_record_to_graph(record)
        assert not torch.isnan(g.x).any()
        assert not torch.isinf(g.x).any()
        assert not torch.isnan(g.edge_attr).any()
        assert not torch.isinf(g.edge_attr).any()


# ---------------------------------------------------------------------------
# TestBuildDataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    """Batch graph construction."""

    def test_dataset_length(self):
        records = [
            _make_corner_record(match_id=1, seed=1),
            _make_corner_record(match_id=2, seed=2),
            _make_corner_record(match_id=3, seed=3),
        ]
        graphs = build_graph_dataset(records, edge_type="knn", k=6)
        assert len(graphs) == 3

    def test_dataset_varied_matches(self):
        records = [
            _make_corner_record(match_id=100, seed=10),
            _make_corner_record(match_id=100, seed=11),
            _make_corner_record(match_id=200, seed=20),
        ]
        graphs = build_graph_dataset(records)
        match_ids = set(g.match_id for g in graphs)
        assert match_ids == {"100", "200"}


# ---------------------------------------------------------------------------
# TestLOMOSplit
# ---------------------------------------------------------------------------

class TestLOMOSplit:
    """Leave-one-match-out split and dataset utilities."""

    def _make_graphs(self):
        """Build a small graph list with 3 matches."""
        records = [
            _make_corner_record(match_id=100, seed=1),
            _make_corner_record(match_id=100, seed=2),
            _make_corner_record(match_id=200, seed=3),
            _make_corner_record(match_id=200, seed=4),
            _make_corner_record(match_id=300, seed=5),
        ]
        return build_graph_dataset(records)

    def test_lomo_split_sizes(self):
        from corner_prediction.data.dataset import lomo_split
        graphs = self._make_graphs()
        train, test = lomo_split(graphs, held_out_match_id="200")
        assert len(train) + len(test) == len(graphs)
        assert len(test) == 2  # match 200 has 2 corners
        assert len(train) == 3

    def test_lomo_split_no_leak(self):
        from corner_prediction.data.dataset import lomo_split
        graphs = self._make_graphs()
        train, test = lomo_split(graphs, held_out_match_id="100")
        train_ids = set(g.match_id for g in train)
        test_ids = set(g.match_id for g in test)
        assert "100" not in train_ids
        assert test_ids == {"100"}

    def test_lomo_split_nonexistent_match(self):
        from corner_prediction.data.dataset import lomo_split
        graphs = self._make_graphs()
        train, test = lomo_split(graphs, held_out_match_id="999")
        assert len(test) == 0
        assert len(train) == len(graphs)

    def test_get_match_ids(self):
        from corner_prediction.data.dataset import get_match_ids
        graphs = self._make_graphs()
        ids = get_match_ids(graphs)
        assert ids == ["100", "200", "300"]

    def test_lomo_split_int_match_id(self):
        """Match ID passed as int should still work (cast to str)."""
        from corner_prediction.data.dataset import lomo_split
        graphs = self._make_graphs()
        train, test = lomo_split(graphs, held_out_match_id=200)
        assert len(test) == 2
