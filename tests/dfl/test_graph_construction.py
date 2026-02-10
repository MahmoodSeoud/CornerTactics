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
