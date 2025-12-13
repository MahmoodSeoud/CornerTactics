"""Tests for GSR post-processing functions.

Tests follow TDD approach: Red -> Green -> Refactor
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestParseStateFile:
    """Tests for parsing .pklz state files."""

    def test_parse_state_file_returns_dataframe(self):
        """parse_state_file should return a DataFrame with expected columns."""
        from scripts.postprocess_gsr import parse_state_file

        state_path = Path('/home/mseo/CornerTactics/outputs/states/CORNER-0000.pklz')
        if not state_path.exists():
            pytest.skip("Test state file not available")

        df = parse_state_file(state_path)

        assert isinstance(df, pd.DataFrame)
        assert 'frame' in df.columns
        assert 'track_id' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'role' in df.columns
        assert 'team' in df.columns

    def test_parse_state_file_extracts_pitch_coords(self):
        """parse_state_file should extract x, y from bbox_pitch dict."""
        from scripts.postprocess_gsr import parse_state_file

        state_path = Path('/home/mseo/CornerTactics/outputs/states/CORNER-0000.pklz')
        if not state_path.exists():
            pytest.skip("Test state file not available")

        df = parse_state_file(state_path)

        # All rows should have valid coordinates (CORNER-0000 has 100% coverage)
        assert df['x'].notna().sum() > 0
        assert df['y'].notna().sum() > 0

        # Coordinates should be in reasonable pitch range
        # Allow some margin beyond standard pitch (105m x 68m) for calibration error
        assert df['x'].min() >= -65  # pitch is 105m, center at 0
        assert df['x'].max() <= 65
        assert df['y'].min() >= -50  # pitch is 68m, center at 0 (corners go beyond sideline)
        assert df['y'].max() <= 50

    def test_parse_state_file_has_frame_numbers(self):
        """parse_state_file should have integer frame numbers."""
        from scripts.postprocess_gsr import parse_state_file

        state_path = Path('/home/mseo/CornerTactics/outputs/states/CORNER-0000.pklz')
        if not state_path.exists():
            pytest.skip("Test state file not available")

        df = parse_state_file(state_path)

        # Frame should be integer or can be cast to int
        assert df['frame'].dtype in [np.int64, np.int32, np.float64]
        if df['frame'].dtype == np.float64:
            # Check no NaN
            assert df['frame'].notna().all()


class TestComputeVelocities:
    """Tests for velocity computation."""

    def test_compute_velocities_adds_columns(self):
        """compute_velocities should add vx, vy, speed columns."""
        from scripts.postprocess_gsr import compute_velocities

        # Simple test data: 2 tracks, 3 frames each
        df = pd.DataFrame({
            'frame': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 10.0, 10.0, 10.0],
            'y': [0.0, 0.0, 0.0, 5.0, 6.0, 7.0],
            'role': ['player'] * 6,
            'team': ['left'] * 6
        })

        result = compute_velocities(df, fps=25.0)

        assert 'vx' in result.columns
        assert 'vy' in result.columns
        assert 'speed' in result.columns

    def test_compute_velocities_correct_values(self):
        """compute_velocities should compute correct velocity values."""
        from scripts.postprocess_gsr import compute_velocities

        # Track 1: moves 1m/frame in x, so at 25fps, vx = 25 m/s
        # Track 2: moves 1m/frame in y, so at 25fps, vy = 25 m/s
        df = pd.DataFrame({
            'frame': [0, 1, 2, 0, 1, 2],
            'track_id': [1, 1, 1, 2, 2, 2],
            'x': [0.0, 1.0, 2.0, 10.0, 10.0, 10.0],
            'y': [0.0, 0.0, 0.0, 5.0, 6.0, 7.0],
            'role': ['player'] * 6,
            'team': ['left'] * 6
        })

        result = compute_velocities(df, fps=25.0)

        # Track 1: vx should be 25 m/s after first frame
        track1 = result[result['track_id'] == 1].sort_values('frame')
        assert track1.iloc[1]['vx'] == pytest.approx(25.0)
        assert track1.iloc[1]['vy'] == pytest.approx(0.0)

        # Track 2: vy should be 25 m/s after first frame
        track2 = result[result['track_id'] == 2].sort_values('frame')
        assert track2.iloc[1]['vx'] == pytest.approx(0.0)
        assert track2.iloc[1]['vy'] == pytest.approx(25.0)

    def test_compute_velocities_first_frame_zero(self):
        """First frame of each track should have zero velocity."""
        from scripts.postprocess_gsr import compute_velocities

        df = pd.DataFrame({
            'frame': [0, 1, 2],
            'track_id': [1, 1, 1],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'role': ['player'] * 3,
            'team': ['left'] * 3
        })

        result = compute_velocities(df, fps=25.0)
        first_row = result[result['frame'] == 0].iloc[0]

        assert first_row['vx'] == 0.0
        assert first_row['vy'] == 0.0
        assert first_row['speed'] == 0.0


class TestExtractSnapshot:
    """Tests for extracting snapshots at specific frames."""

    def test_extract_snapshot_at_target_frame(self):
        """extract_snapshot should return rows at target frame."""
        from scripts.postprocess_gsr import extract_snapshot

        df = pd.DataFrame({
            'frame': [48, 49, 50, 51, 52] * 3,
            'track_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            'x': list(range(15)),
            'y': list(range(15)),
            'role': ['player'] * 15,
            'team': ['left'] * 15
        })

        snapshot = extract_snapshot(df, target_frame=50)

        assert len(snapshot) == 3  # 3 tracks at frame 50
        assert (snapshot['frame'] == 50).all()

    def test_extract_snapshot_closest_frame(self):
        """extract_snapshot should find closest frame if target not available."""
        from scripts.postprocess_gsr import extract_snapshot

        # No frame 50, but has frame 48 and 52
        df = pd.DataFrame({
            'frame': [48, 52],
            'track_id': [1, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 1.0],
            'role': ['player'] * 2,
            'team': ['left'] * 2
        })

        snapshot = extract_snapshot(df, target_frame=50)

        # Should pick 48 (distance 2) or 52 (distance 2)
        assert len(snapshot) == 1
        assert snapshot['frame'].iloc[0] in [48, 52]
