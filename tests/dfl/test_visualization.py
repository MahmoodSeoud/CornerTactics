"""Tests for corner kick visualization.

Following TDD, these tests are written first to define the expected behavior.
"""

import pytest
from pathlib import Path
import tempfile

# Test data paths
METRICA_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "metrica" / "data"
DFL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dfl"


class TestPlotCornerFrame:
    """Tests for plotting a single corner kick frame."""

    def test_plot_corner_frame_returns_figure(self):
        """plot_corner_frame should return a matplotlib figure."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.visualization import plot_corner_frame

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
        fig = plot_corner_frame(
            frame=frames[mid_idx],
            velocities=velocities[mid_idx],
            corner_event=corners[0],
        )

        import matplotlib.pyplot as plt

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_corner_frame_can_save_to_file(self):
        """plot_corner_frame should be able to save to a file."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
        )
        from src.dfl.visualization import plot_corner_frame

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

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            fig = plot_corner_frame(
                frame=frames[mid_idx],
                velocities=velocities[mid_idx],
                corner_event=corners[0],
                save_path=output_path,
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

            import matplotlib.pyplot as plt
            plt.close(fig)
        finally:
            output_path.unlink(missing_ok=True)


class TestVisualizationCoordinates:
    """Tests for coordinate system handling in visualization."""

    def test_coordinates_within_pitch_bounds(self):
        """Player positions should be within standard pitch dimensions."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
        )

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

        # Check coordinates are reasonable
        for frame in frames[:10]:
            for player_id, pdata in frame.players_data.items():
                if pdata.coordinates:
                    x = pdata.coordinates.x
                    y = pdata.coordinates.y
                    # Standard pitch: 105m x 68m, but coordinates may vary
                    # Just check they're not wildly wrong
                    assert -10 <= x <= 120, f"X={x} out of bounds"
                    assert -10 <= y <= 80, f"Y={y} out of bounds"
