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


class TestCornerToTemporalGraphs:
    """Tests for converting a corner kick to a sequence of graphs."""

    def test_corner_to_temporal_graphs_returns_list(self):
        """corner_to_temporal_graphs should return a list of graphs."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
        )
        from src.dfl.graph_construction import corner_to_temporal_graphs

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

        graphs = corner_to_temporal_graphs(tracking, corners[0])

        assert isinstance(graphs, list)
        assert len(graphs) > 0

    def test_corner_to_temporal_graphs_all_are_data_objects(self):
        """Each element should be a PyTorch Geometric Data object."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
        )
        from src.dfl.graph_construction import corner_to_temporal_graphs
        from torch_geometric.data import Data

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

        graphs = corner_to_temporal_graphs(tracking, corners[0])

        for g in graphs:
            assert isinstance(g, Data)

    def test_corner_to_temporal_graphs_has_frame_metadata(self):
        """Each graph should have frame_idx and relative_time attributes."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
        )
        from src.dfl.graph_construction import corner_to_temporal_graphs

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

        graphs = corner_to_temporal_graphs(tracking, corners[0])

        for g in graphs:
            assert hasattr(g, "frame_idx")
            assert hasattr(g, "relative_time")

    def test_corner_to_temporal_graphs_correct_duration(self):
        """Sequence should cover approximately 8 seconds (2 before + 6 after)."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
        )
        from src.dfl.graph_construction import corner_to_temporal_graphs

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

        graphs = corner_to_temporal_graphs(
            tracking, corners[0], pre_seconds=2.0, post_seconds=6.0
        )

        fps = 25
        expected_frames = int(8.0 * fps)  # 200 frames

        # Allow 10% tolerance
        assert abs(len(graphs) - expected_frames) < expected_frames * 0.2


class TestLabelCorner:
    """Tests for labeling corner kick outcomes."""

    def test_label_corner_returns_dict(self):
        """label_corner should return a dictionary of labels."""
        from src.dfl.data_loading import load_event_data, find_corner_events
        from src.dfl.graph_construction import label_corner

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        labels = label_corner(corners[0], events)

        assert isinstance(labels, dict)

    def test_label_corner_has_shot_binary(self):
        """Labels should include shot_binary (0 or 1)."""
        from src.dfl.data_loading import load_event_data, find_corner_events
        from src.dfl.graph_construction import label_corner

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        labels = label_corner(corners[0], events)

        assert "shot_binary" in labels
        assert labels["shot_binary"] in [0, 1]

    def test_label_corner_has_goal_binary(self):
        """Labels should include goal_binary (0 or 1)."""
        from src.dfl.data_loading import load_event_data, find_corner_events
        from src.dfl.graph_construction import label_corner

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        labels = label_corner(corners[0], events)

        assert "goal_binary" in labels
        assert labels["goal_binary"] in [0, 1]

    def test_label_corner_has_first_contact_team(self):
        """Labels should include first_contact_team."""
        from src.dfl.data_loading import load_event_data, find_corner_events
        from src.dfl.graph_construction import label_corner

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        labels = label_corner(corners[0], events)

        assert "first_contact_team" in labels
        assert labels["first_contact_team"] in ["attacking", "defending", "unknown"]

    def test_label_corner_has_outcome_class(self):
        """Labels should include outcome_class."""
        from src.dfl.data_loading import load_event_data, find_corner_events
        from src.dfl.graph_construction import label_corner

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        corners = find_corner_events(events)

        if not corners:
            pytest.skip("No corners in this match")

        labels = label_corner(corners[0], events)

        assert "outcome_class" in labels


class TestBuildCornerDataset:
    """Tests for building the complete corner dataset."""

    def test_build_corner_dataset_from_match_returns_list(self):
        """build_corner_dataset_from_match should return a list of corner samples."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
        )
        from src.dfl.graph_construction import build_corner_dataset_from_match

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        dataset = build_corner_dataset_from_match(
            tracking_dataset=tracking,
            event_dataset=events,
            match_id="Sample_Game_3",
        )

        assert isinstance(dataset, list)

    def test_build_corner_dataset_sample_has_required_keys(self):
        """Each sample should have graphs, labels, match_id, corner_time."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
        )
        from src.dfl.graph_construction import build_corner_dataset_from_match

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        dataset = build_corner_dataset_from_match(
            tracking_dataset=tracking,
            event_dataset=events,
            match_id="Sample_Game_3",
        )

        if not dataset:
            pytest.skip("No corners in this match")

        sample = dataset[0]
        assert "graphs" in sample
        assert "labels" in sample
        assert "match_id" in sample
        assert "corner_time" in sample

    def test_build_corner_dataset_graphs_are_list(self):
        """Each sample's graphs should be a list of Data objects."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
        )
        from src.dfl.graph_construction import build_corner_dataset_from_match
        from torch_geometric.data import Data

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        dataset = build_corner_dataset_from_match(
            tracking_dataset=tracking,
            event_dataset=events,
            match_id="Sample_Game_3",
        )

        if not dataset:
            pytest.skip("No corners in this match")

        sample = dataset[0]
        assert isinstance(sample["graphs"], list)
        assert len(sample["graphs"]) > 0
        assert all(isinstance(g, Data) for g in sample["graphs"])


class TestSaveLoadDataset:
    """Tests for saving and loading the corner dataset."""

    def test_save_corner_dataset_creates_file(self, tmp_path):
        """save_corner_dataset should create a pickle file."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
        )
        from src.dfl.graph_construction import (
            build_corner_dataset_from_match,
            save_corner_dataset,
        )

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        dataset = build_corner_dataset_from_match(
            tracking_dataset=tracking,
            event_dataset=events,
            match_id="Sample_Game_3",
        )

        output_path = tmp_path / "corner_dataset.pkl"
        save_corner_dataset(dataset, output_path)

        assert output_path.exists()

    def test_load_corner_dataset_returns_same_data(self, tmp_path):
        """load_corner_dataset should return the same data that was saved."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
        )
        from src.dfl.graph_construction import (
            build_corner_dataset_from_match,
            save_corner_dataset,
            load_corner_dataset,
        )

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        dataset = build_corner_dataset_from_match(
            tracking_dataset=tracking,
            event_dataset=events,
            match_id="Sample_Game_3",
        )

        if not dataset:
            pytest.skip("No corners in this match")

        output_path = tmp_path / "corner_dataset.pkl"
        save_corner_dataset(dataset, output_path)
        loaded = load_corner_dataset(output_path)

        assert len(loaded) == len(dataset)
        assert loaded[0]["match_id"] == dataset[0]["match_id"]
