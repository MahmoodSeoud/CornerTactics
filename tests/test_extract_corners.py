"""
Tests for corner kick extraction with freeze frames.

Following TDD approach for Task 1 of PLAN.md:
Extract corners from StatsBomb event data and match with freeze frames.
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import from the script (handle the 01_ prefix)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_corners",
    Path(__file__).parent.parent / "scripts" / "01_extract_corners_with_freeze_frames.py"
)
extract_corners = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_corners)

extract_corners_from_match = extract_corners.extract_corners_from_match
match_corners_with_freeze_frames = extract_corners.match_corners_with_freeze_frames
process_all_matches = extract_corners.process_all_matches
is_corner_kick = extract_corners.is_corner_kick


class TestCornerExtraction:
    """Test corner kick extraction functionality."""

    def test_is_corner_kick_identifies_corner(self):
        """Test that corner kicks are correctly identified."""
        corner_event = {
            "id": "test-uuid",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
        }
        assert is_corner_kick(corner_event) is True

    def test_is_corner_kick_rejects_normal_pass(self):
        """Test that normal passes are not identified as corners."""
        normal_pass = {
            "id": "test-uuid",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Recovery"}},
        }
        assert is_corner_kick(normal_pass) is False

    def test_is_corner_kick_rejects_non_pass(self):
        """Test that non-pass events are not identified as corners."""
        shot_event = {
            "id": "test-uuid",
            "type": {"name": "Shot"},
        }
        assert is_corner_kick(shot_event) is False

    def test_extract_corners_from_match_returns_list(self):
        """Test that extract_corners_from_match returns a list."""
        # Use a real match file from the dataset
        match_id = "15946"
        corners = extract_corners_from_match(match_id)

        assert isinstance(corners, list)
        assert len(corners) > 0  # We know this match has corners

        # Verify each corner has required fields
        for corner in corners:
            assert "id" in corner
            assert "type" in corner
            assert corner["type"]["name"] == "Pass"
            assert corner["pass"]["type"]["name"] == "Corner"

    def test_match_corners_with_freeze_frames_returns_matched_data(self):
        """Test that corners are matched with freeze frames correctly."""
        # Create sample corner and freeze frame data
        corner_event = {
            "id": "test-corner-uuid",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
            "location": [120.0, 80.0],
        }

        freeze_frames = [
            {
                "event_uuid": "other-uuid",
                "freeze_frame": [{"location": [50, 40]}],
            },
            {
                "event_uuid": "test-corner-uuid",
                "freeze_frame": [{"location": [60, 40]}],
            },
        ]

        matched = match_corners_with_freeze_frames(
            [corner_event], freeze_frames, "test_match"
        )

        assert len(matched) == 1
        assert matched[0]["match_id"] == "test_match"
        assert matched[0]["event"]["id"] == "test-corner-uuid"
        assert len(matched[0]["freeze_frame"]) == 1

    def test_match_corners_with_freeze_frames_handles_no_match(self):
        """Test that corners without freeze frames are excluded."""
        corner_event = {
            "id": "unmatched-corner-uuid",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
        }

        freeze_frames = [
            {"event_uuid": "other-uuid", "freeze_frame": []},
        ]

        matched = match_corners_with_freeze_frames(
            [corner_event], freeze_frames, "test_match"
        )

        assert len(matched) == 0

    def test_process_all_matches_creates_output_file(self, tmp_path):
        """Test that process_all_matches creates the output file."""
        output_file = tmp_path / "test_output.json"

        # Process with limit for testing
        result = process_all_matches(
            output_path=str(output_file),
            limit=5  # Only process first 5 matches
        )

        assert output_file.exists()
        assert result["total_matches_processed"] > 0
        assert result["total_corners_with_freeze_frames"] >= 0

        # Verify file contents
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)

    def test_output_format_matches_specification(self):
        """Test that output matches the required format from PLAN.md."""
        # Process one match that has both events and freeze frames
        match_id = "15946"
        corners = extract_corners_from_match(match_id)

        # Load freeze frames
        freeze_frame_path = Path("data/statsbomb/freeze-frames") / f"{match_id}.json"
        if freeze_frame_path.exists():
            with open(freeze_frame_path) as f:
                freeze_frames = json.load(f)

            matched = match_corners_with_freeze_frames(
                corners, freeze_frames, match_id
            )

            if len(matched) > 0:
                sample = matched[0]

                # Verify format
                assert "match_id" in sample
                assert "event" in sample
                assert "freeze_frame" in sample

                # Verify event is complete corner object
                assert isinstance(sample["event"], dict)
                assert "id" in sample["event"]

                # Verify freeze_frame is array
                assert isinstance(sample["freeze_frame"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
