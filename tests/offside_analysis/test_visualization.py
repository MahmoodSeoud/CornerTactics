"""Tests for offside visualization module.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


@pytest.fixture
def sample_corners():
    """Sample corners with different shot outcomes for testing."""
    return [
        # Shot corners (shot_outcome=1) - attackers more aggressive
        {
            "match_id": "1",
            "event": {"id": "c1", "location": [120.0, 0.0]},
            "freeze_frame": [
                {"location": [112.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [110.0, 40.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [108.0, 45.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [120.0, 0.0], "teammate": True, "keeper": False, "actor": True},
                {"location": [108.0, 38.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [107.0, 42.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},
            ],
            "shot_outcome": 1,
        },
        {
            "match_id": "2",
            "event": {"id": "c2", "location": [120.0, 80.0]},
            "freeze_frame": [
                {"location": [114.0, 38.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [111.0, 42.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [109.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [120.0, 80.0], "teammate": True, "keeper": False, "actor": True},
                {"location": [110.0, 40.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [109.0, 36.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},
            ],
            "shot_outcome": 1,
        },
        # No-shot corners (shot_outcome=0) - attackers more conservative
        {
            "match_id": "3",
            "event": {"id": "c3", "location": [120.0, 0.0]},
            "freeze_frame": [
                {"location": [105.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [103.0, 40.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [106.0, 45.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [120.0, 0.0], "teammate": True, "keeper": False, "actor": True},
                {"location": [110.0, 38.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [111.0, 42.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},
            ],
            "shot_outcome": 0,
        },
        {
            "match_id": "4",
            "event": {"id": "c4", "location": [120.0, 80.0]},
            "freeze_frame": [
                {"location": [104.0, 38.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [102.0, 42.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [105.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [120.0, 80.0], "teammate": True, "keeper": False, "actor": True},
                {"location": [112.0, 40.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [113.0, 36.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},
            ],
            "shot_outcome": 0,
        },
    ]


class TestAveragePositionPlot:
    """Test average player position plotting."""

    def test_compute_average_positions_by_outcome(self, sample_corners):
        """Should compute average positions grouped by outcome."""
        from experiments.offside_analysis.visualization import compute_average_positions

        avg_positions = compute_average_positions(sample_corners)

        assert 'shot' in avg_positions
        assert 'no_shot' in avg_positions
        assert 'attackers' in avg_positions['shot']
        assert 'defenders' in avg_positions['shot']

    def test_average_positions_shape(self, sample_corners):
        """Average positions should have correct shape."""
        from experiments.offside_analysis.visualization import compute_average_positions

        avg_positions = compute_average_positions(sample_corners)

        # Should have x, y coordinates for each player type
        assert avg_positions['shot']['attackers'].shape[1] == 2  # x, y
        assert len(avg_positions['shot']['attackers']) > 0

    def test_plot_average_positions_returns_figure(self, sample_corners):
        """Should return matplotlib figure."""
        from experiments.offside_analysis.visualization import plot_average_positions

        fig = plot_average_positions(sample_corners)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestHeatmaps:
    """Test heatmap visualization."""

    def test_create_position_heatmap(self, sample_corners):
        """Should create position density heatmap."""
        from experiments.offside_analysis.visualization import create_position_heatmap

        fig = create_position_heatmap(sample_corners, player_type='attackers')

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_difference_heatmap(self, sample_corners):
        """Should create heatmap showing positional differences between outcomes."""
        from experiments.offside_analysis.visualization import create_difference_heatmap

        fig = create_difference_heatmap(sample_corners)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPitchDrawing:
    """Test football pitch drawing utilities."""

    def test_draw_pitch_returns_axes(self):
        """Should return axes with pitch drawn."""
        from experiments.offside_analysis.visualization import draw_pitch

        fig, ax = plt.subplots()
        ax = draw_pitch(ax)

        assert ax is not None
        plt.close(fig)

    def test_draw_half_pitch(self):
        """Should draw only the attacking half of pitch."""
        from experiments.offside_analysis.visualization import draw_half_pitch

        fig, ax = plt.subplots()
        ax = draw_half_pitch(ax)

        assert ax is not None
        plt.close(fig)


class TestFeatureDistributionPlots:
    """Test feature distribution visualization."""

    def test_plot_feature_distributions(self, sample_corners):
        """Should plot feature distributions by outcome."""
        from experiments.offside_analysis.visualization import plot_feature_distributions

        fig = plot_feature_distributions(sample_corners)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_single_feature(self, sample_corners):
        """Should plot single feature distribution."""
        from experiments.offside_analysis.visualization import plot_single_feature

        fig = plot_single_feature(
            sample_corners,
            feature_name='attacker_defender_gap'
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCornerVisualization:
    """Test single corner visualization."""

    def test_plot_single_corner(self, sample_corners):
        """Should visualize a single corner freeze frame."""
        from experiments.offside_analysis.visualization import plot_single_corner

        fig = plot_single_corner(sample_corners[0])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_corner_with_offside_line(self, sample_corners):
        """Should show offside line in visualization."""
        from experiments.offside_analysis.visualization import plot_single_corner

        fig = plot_single_corner(sample_corners[0], show_offside_line=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
