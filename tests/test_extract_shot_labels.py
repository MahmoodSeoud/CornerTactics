"""
Tests for scripts/07_extract_shot_labels.py

Following TDD approach for binary shot label extraction.
"""

import json
import pytest
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the module being tested
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_shot_labels",
    Path(__file__).parent.parent / "scripts" / "07_extract_shot_labels.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Import functions from the module
find_event_index = module.find_event_index
check_shot_in_lookahead = module.check_shot_in_lookahead
add_shot_label = module.add_shot_label
load_corners = module.load_corners
load_match_events = module.load_match_events
save_labeled_corners = module.save_labeled_corners
validate_distribution = module.validate_distribution


# Test fixtures
@pytest.fixture
def sample_corners():
    """Sample corners_with_freeze_frames.json data"""
    return [
        {
            "match_id": "123456",
            "event": {
                "id": "corner-uuid-1",
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Corner"}},
                "location": [100, 20],
                "period": 1,
                "minute": 15
            },
            "freeze_frame": [
                {"location": [110, 40], "teammate": True},
                {"location": [115, 40], "teammate": False}
            ]
        },
        {
            "match_id": "123456",
            "event": {
                "id": "corner-uuid-2",
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Corner"}},
                "location": [100, 60],
                "period": 2,
                "minute": 75
            },
            "freeze_frame": [
                {"location": [110, 40], "teammate": True}
            ]
        }
    ]


@pytest.fixture
def match_events_with_shot():
    """Sample match events with shot in lookahead window"""
    return [
        {
            "id": "pre-corner-event",
            "type": {"name": "Pass"},
            "minute": 14
        },
        {
            "id": "corner-uuid-1",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
            "minute": 15
        },
        {
            "id": "event-1",
            "type": {"name": "Ball Receipt*"},
            "minute": 15
        },
        {
            "id": "event-2",
            "type": {"name": "Pass"},
            "minute": 15
        },
        {
            "id": "event-3",
            "type": {"name": "Shot"},  # Shot within lookahead window
            "minute": 15
        },
        {
            "id": "event-4",
            "type": {"name": "Goal Keeper"},
            "minute": 15
        }
    ]


@pytest.fixture
def match_events_no_shot():
    """Sample match events without shot in lookahead window"""
    return [
        {
            "id": "corner-uuid-2",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
            "minute": 75
        },
        {
            "id": "event-1",
            "type": {"name": "Clearance"},
            "minute": 75
        },
        {
            "id": "event-2",
            "type": {"name": "Ball Recovery"},
            "minute": 75
        },
        {
            "id": "event-3",
            "type": {"name": "Pass"},
            "minute": 76
        },
        {
            "id": "event-4",
            "type": {"name": "Duel"},
            "minute": 76
        },
        {
            "id": "event-5",
            "type": {"name": "Pressure"},
            "minute": 76
        }
    ]


@pytest.fixture
def temp_data_dirs(tmp_path):
    """Create temporary data directories"""
    processed_dir = tmp_path / "data" / "processed"
    events_dir = tmp_path / "data" / "statsbomb" / "events" / "events"
    processed_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    return {
        "processed": processed_dir,
        "events": events_dir,
        "root": tmp_path
    }


class TestShotDetection:
    """Test shot detection in lookahead window"""

    def test_detect_shot_in_window(self, match_events_with_shot):
        """RED: Should detect shot within 5-event lookahead window"""
        corner_index = 1  # corner-uuid-1
        window_size = 5
        has_shot = check_shot_in_lookahead(
            match_events_with_shot, corner_index, window_size
        )
        assert has_shot is True

    def test_no_shot_in_window(self, match_events_no_shot):
        """RED: Should return False when no shot in lookahead window"""
        corner_index = 0  # corner-uuid-2
        window_size = 5
        has_shot = check_shot_in_lookahead(
            match_events_no_shot, corner_index, window_size
        )
        assert has_shot is False

    def test_shot_beyond_window(self):
        """RED: Should not detect shot beyond window size"""
        events = [
            {"id": "corner", "type": {"name": "Pass"}},
            {"id": "e1", "type": {"name": "Ball Receipt*"}},
            {"id": "e2", "type": {"name": "Pass"}},
            {"id": "e3", "type": {"name": "Duel"}},
            {"id": "e4", "type": {"name": "Clearance"}},
            {"id": "e5", "type": {"name": "Pass"}},
            {"id": "e6", "type": {"name": "Shot"}},  # Beyond 5-event window
        ]
        corner_index = 0
        window_size = 5
        has_shot = check_shot_in_lookahead(events, corner_index, window_size)
        assert has_shot is False

    def test_shot_at_window_boundary(self):
        """RED: Should detect shot exactly at window boundary (5th event)"""
        events = [
            {"id": "corner", "type": {"name": "Pass"}},
            {"id": "e1", "type": {"name": "Ball Receipt*"}},
            {"id": "e2", "type": {"name": "Pass"}},
            {"id": "e3", "type": {"name": "Duel"}},
            {"id": "e4", "type": {"name": "Clearance"}},
            {"id": "e5", "type": {"name": "Shot"}},  # Exactly at window boundary
        ]
        corner_index = 0
        window_size = 5
        has_shot = check_shot_in_lookahead(events, corner_index, window_size)
        assert has_shot is True

    def test_corner_near_end_of_match(self):
        """RED: Should handle corner with fewer than 5 events remaining"""
        events = [
            {"id": "corner", "type": {"name": "Pass"}},
            {"id": "e1", "type": {"name": "Ball Receipt*"}},
            {"id": "e2", "type": {"name": "Shot"}},
        ]
        corner_index = 0
        window_size = 5
        has_shot = check_shot_in_lookahead(events, corner_index, window_size)
        assert has_shot is True


class TestLabelExtraction:
    """Test full label extraction pipeline"""

    def test_add_shot_label_positive(self, sample_corners, match_events_with_shot):
        """RED: Should add shot_outcome=1 when shot detected"""
        corner = sample_corners[0]
        labeled = add_shot_label(corner, match_events_with_shot, window_size=5)

        assert "shot_outcome" in labeled
        assert labeled["shot_outcome"] == 1

    def test_add_shot_label_negative(self, sample_corners, match_events_no_shot):
        """RED: Should add shot_outcome=0 when no shot detected"""
        corner = sample_corners[1]
        labeled = add_shot_label(corner, match_events_no_shot, window_size=5)

        assert "shot_outcome" in labeled
        assert labeled["shot_outcome"] == 0

    def test_handle_missing_corner_in_events(self, sample_corners):
        """RED: Should handle case where corner UUID not found in events"""
        events = [
            {"id": "other-event", "type": {"name": "Pass"}}
        ]
        corner = sample_corners[0]
        labeled = add_shot_label(corner, events, window_size=5)

        assert "shot_outcome" in labeled
        assert labeled["shot_outcome"] == 0  # Default to no shot

    def test_load_corners_from_file(self, temp_data_dirs, sample_corners):
        """RED: Should load corners_with_freeze_frames.json"""
        corners_file = temp_data_dirs["processed"] / "corners_with_freeze_frames.json"
        with open(corners_file, 'w') as f:
            json.dump(sample_corners, f)

        loaded = load_corners(corners_file)
        assert len(loaded) == 2
        assert loaded[0]["match_id"] == "123456"


class TestOutputValidation:
    """Test output file creation and validation"""

    def test_save_labeled_corners(self, temp_data_dirs, sample_corners):
        """RED: Should save corners with shot labels to output file"""
        labeled = [
            {**corner, "shot_outcome": 1}
            for corner in sample_corners
        ]

        output_file = temp_data_dirs["processed"] / "corners_with_shot_labels.json"
        save_labeled_corners(labeled, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert all("shot_outcome" in corner for corner in data)

    def test_validate_class_distribution(self):
        """RED: Should calculate shot vs no-shot distribution"""
        # Simulate 15% shot rate (expected from PLAN.md)
        labeled_corners = [
            {"shot_outcome": 1} for _ in range(15)
        ] + [
            {"shot_outcome": 0} for _ in range(85)
        ]

        distribution = validate_distribution(labeled_corners)

        assert distribution["shot_percentage"] == pytest.approx(15.0, abs=0.1)
        assert distribution["no_shot_percentage"] == pytest.approx(85.0, abs=0.1)
        assert distribution["imbalance_ratio"] == pytest.approx(85/15, abs=0.1)

    def test_distribution_handles_edge_cases(self):
        """RED: Should handle edge cases (all shots, no shots)"""
        # All shots
        all_shots = [{"shot_outcome": 1} for _ in range(10)]
        dist = validate_distribution(all_shots)
        assert dist["shot_percentage"] == 100.0

        # No shots
        no_shots = [{"shot_outcome": 0} for _ in range(10)]
        dist = validate_distribution(no_shots)
        assert dist["shot_percentage"] == 0.0
