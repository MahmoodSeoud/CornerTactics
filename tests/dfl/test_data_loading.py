"""Tests for DFL/Metrica tracking data loading.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import pytest
from pathlib import Path

# Test data paths
METRICA_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "metrica" / "data"
DFL_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dfl"


class TestLoadTrackingData:
    """Tests for loading tracking data from different providers."""

    def test_load_metrica_tracking_returns_dataset(self):
        """Loading Metrica tracking data should return a kloppy dataset."""
        from src.dfl.data_loading import load_tracking_data

        dataset = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        assert dataset is not None
        assert hasattr(dataset, "records")
        assert len(dataset.records) > 0

    def test_load_metrica_tracking_has_correct_frame_rate(self):
        """Metrica tracking data should be at 25fps."""
        from src.dfl.data_loading import load_tracking_data

        dataset = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        assert dataset.metadata.frame_rate == 25

    def test_load_metrica_tracking_has_players(self):
        """Each frame should have data for players (22+ for Metrica which includes subs)."""
        from src.dfl.data_loading import load_tracking_data

        dataset = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        # Check first frame
        frame = dataset.records[0]
        player_count = len(frame.players_data)
        # Metrica includes substitutes (35 total), DFL will have 22
        assert player_count >= 22, f"Expected at least 22 players, got {player_count}"

    def test_load_metrica_tracking_has_ball_coordinates(self):
        """Tracking data should include ball position."""
        from src.dfl.data_loading import load_tracking_data

        dataset = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        # Find a frame with ball data (ball may not be visible in all frames)
        ball_found = False
        for frame in dataset.records[:100]:
            if frame.ball_coordinates is not None:
                ball_found = True
                assert hasattr(frame.ball_coordinates, "x")
                assert hasattr(frame.ball_coordinates, "y")
                break

        assert ball_found, "No ball coordinates found in first 100 frames"


class TestLoadEventData:
    """Tests for loading event data."""

    def test_load_metrica_events_returns_dataset(self):
        """Loading Metrica event data should return a kloppy dataset."""
        from src.dfl.data_loading import load_event_data

        dataset = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        assert dataset is not None
        assert hasattr(dataset, "events")
        assert len(dataset.events) > 0

    def test_load_metrica_events_have_timestamps(self):
        """Each event should have a timestamp."""
        from src.dfl.data_loading import load_event_data

        dataset = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        for event in dataset.events[:10]:
            assert hasattr(event, "timestamp")


class TestFindCornerEvents:
    """Tests for identifying corner kick events."""

    def test_find_corners_returns_list(self):
        """find_corner_events should return a list of corner kick events."""
        from src.dfl.data_loading import load_event_data, find_corner_events

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        corners = find_corner_events(events)

        assert isinstance(corners, list)

    def test_find_corners_each_has_timestamp(self):
        """Each corner event should have a timestamp."""
        from src.dfl.data_loading import load_event_data, find_corner_events

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        corners = find_corner_events(events)

        for corner in corners:
            assert hasattr(corner, "timestamp")

    def test_find_corners_each_has_team(self):
        """Each corner event should identify the attacking team."""
        from src.dfl.data_loading import load_event_data, find_corner_events

        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        corners = find_corner_events(events)

        for corner in corners:
            assert hasattr(corner, "team")


class TestExtractCornerSequence:
    """Tests for extracting tracking data around corner kicks."""

    def test_extract_corner_sequence_returns_frames(self):
        """Extracting a corner sequence should return a list of frames."""
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

        frames = extract_corner_sequence(
            tracking_dataset=tracking,
            corner_event=corners[0],
            pre_seconds=2.0,
            post_seconds=6.0,
        )

        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_extract_corner_sequence_correct_duration(self):
        """Extracted sequence should cover approximately 8 seconds (2+6)."""
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

        frames = extract_corner_sequence(
            tracking_dataset=tracking,
            corner_event=corners[0],
            pre_seconds=2.0,
            post_seconds=6.0,
        )

        fps = tracking.metadata.frame_rate
        expected_frames = int(8.0 * fps)  # 8 seconds * 25fps = 200 frames

        # Allow 10% tolerance
        assert abs(len(frames) - expected_frames) < expected_frames * 0.1


class TestComputeVelocities:
    """Tests for computing velocity vectors from position data."""

    def test_compute_velocities_returns_dict(self):
        """compute_velocities should return velocities for each player."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
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

        frames = extract_corner_sequence(
            tracking_dataset=tracking,
            corner_event=corners[0],
            pre_seconds=2.0,
            post_seconds=6.0,
        )

        velocities = compute_velocities(frames, fps=25)

        assert isinstance(velocities, list)
        assert len(velocities) == len(frames)

    def test_compute_velocities_has_vx_vy(self):
        """Each velocity entry should have vx and vy per player."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
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

        frames = extract_corner_sequence(
            tracking_dataset=tracking,
            corner_event=corners[0],
            pre_seconds=2.0,
            post_seconds=6.0,
        )

        velocities = compute_velocities(frames, fps=25)

        # Check middle frame (first and last have edge effects)
        mid_idx = len(velocities) // 2
        for player_id, vel in velocities[mid_idx].items():
            assert "vx" in vel, f"Missing vx for player {player_id}"
            assert "vy" in vel, f"Missing vy for player {player_id}"

    def test_compute_velocities_reasonable_magnitude(self):
        """Velocities should be in reasonable range (0-12 m/s)."""
        from src.dfl.data_loading import (
            load_tracking_data,
            load_event_data,
            find_corner_events,
            extract_corner_sequence,
            compute_velocities,
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

        frames = extract_corner_sequence(
            tracking_dataset=tracking,
            corner_event=corners[0],
            pre_seconds=2.0,
            post_seconds=6.0,
        )

        velocities = compute_velocities(frames, fps=25)

        # Check velocities are reasonable
        for frame_vel in velocities[1:-1]:  # Skip first/last
            for player_id, vel in frame_vel.items():
                speed = (vel["vx"] ** 2 + vel["vy"] ** 2) ** 0.5
                assert speed < 15, f"Unreasonable speed {speed:.1f} m/s for player {player_id}"
