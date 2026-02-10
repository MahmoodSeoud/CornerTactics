"""Tests for Phase 2: Graph Construction Pipeline.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import pytest
import numpy as np
from pathlib import Path

# Test data paths
METRICA_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "metrica" / "data"


class TestFrameToGraph:
    """Tests for converting a single tracking frame to a graph."""

    def test_frame_to_graph_returns_torch_geometric_data(self):
        """frame_to_graph should return a PyTorch Geometric Data object."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.graph_construction import frame_to_graph

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        frames = extract_corner_sequence(tracking, corners[0])
        velocities = compute_velocities(frames, fps=25)

        # Use middle frame
        mid_idx = len(frames) // 2
        graph = frame_to_graph(
            frame=frames[mid_idx],
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        from torch_geometric.data import Data

        assert isinstance(graph, Data)

    def test_frame_to_graph_has_correct_node_features(self):
        """Graph should have 8 features per node: x, y, vx, vy, team, kicker, dist_goal, dist_ball."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.graph_construction import frame_to_graph

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        frames = extract_corner_sequence(tracking, corners[0])
        velocities = compute_velocities(frames, fps=25)

        mid_idx = len(frames) // 2
        graph = frame_to_graph(
            frame=frames[mid_idx],
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        assert graph.x.shape[1] == 8, f"Expected 8 features, got {graph.x.shape[1]}"

    def test_frame_to_graph_has_players_plus_ball(self):
        """Graph should have num_players + 1 (ball) nodes."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.graph_construction import frame_to_graph

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        frames = extract_corner_sequence(tracking, corners[0])
        velocities = compute_velocities(frames, fps=25)

        mid_idx = len(frames) // 2
        frame = frames[mid_idx]

        # Count players with valid coordinates
        n_players = sum(
            1 for p in frame.players_data.values() if p.coordinates is not None
        )

        graph = frame_to_graph(
            frame=frame,
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        # num_nodes should be n_players + 1 (ball)
        assert graph.x.shape[0] == n_players + 1

    def test_frame_to_graph_has_edges(self):
        """Graph should have edges (kNN, marking, ball edges)."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.graph_construction import frame_to_graph

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        frames = extract_corner_sequence(tracking, corners[0])
        velocities = compute_velocities(frames, fps=25)

        mid_idx = len(frames) // 2
        graph = frame_to_graph(
            frame=frames[mid_idx],
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        assert graph.edge_index.shape[0] == 2  # Edge index should be 2 x num_edges
        assert graph.edge_index.shape[1] > 0  # Should have at least some edges

    def test_frame_to_graph_has_position_tensor(self):
        """Graph should have a pos tensor for node positions."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.graph_construction import frame_to_graph

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        frames = extract_corner_sequence(tracking, corners[0])
        velocities = compute_velocities(frames, fps=25)

        mid_idx = len(frames) // 2
        graph = frame_to_graph(
            frame=frames[mid_idx],
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        assert hasattr(graph, "pos")
        assert graph.pos.shape[0] == graph.x.shape[0]  # Same number of nodes
        assert graph.pos.shape[1] == 2  # 2D positions (x, y)
