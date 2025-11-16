"""
Tests for scripts/02_extract_outcome_labels.py

Following TDD approach for corner outcome label extraction.
"""

import json
import pytest
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the module being tested (renamed to 02_extract_outcome_labels.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_outcome_labels",
    Path(__file__).parent.parent / "scripts" / "02_extract_outcome_labels.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Import functions from the module
map_event_to_outcome = module.map_event_to_outcome
find_event_index = module.find_event_index
get_next_event = module.get_next_event
load_corners = module.load_corners
load_match_events = module.load_match_events
add_outcome_label = module.add_outcome_label
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
def sample_match_events():
    """Sample match events with corners and following events"""
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
            "id": "next-event-1",
            "type": {"name": "Ball Receipt*"},
            "minute": 15
        },
        {
            "id": "corner-uuid-2",
            "type": {"name": "Pass"},
            "pass": {"type": {"name": "Corner"}},
            "minute": 75
        },
        {
            "id": "next-event-2",
            "type": {"name": "Clearance"},
            "minute": 75
        }
    ]


@pytest.fixture
def temp_data_dirs(tmp_path):
    """Create temporary data directories"""
    processed_dir = tmp_path / "data" / "processed"
    events_dir = tmp_path / "data" / "statsbomb" / "events"
    processed_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    return {
        "processed": processed_dir,
        "events": events_dir,
        "root": tmp_path
    }


class TestOutcomeMapping:
    """Test outcome type mapping logic"""

    def test_ball_receipt_maps_correctly(self):
        """RED: Ball Receipt* should map to 'Ball Receipt'"""
        event = {"type": {"name": "Ball Receipt*"}}
        outcome = map_event_to_outcome(event)
        assert outcome == "Ball Receipt"

    def test_clearance_maps_correctly(self):
        """RED: Clearance should map to 'Clearance'"""
        event = {"type": {"name": "Clearance"}}
        outcome = map_event_to_outcome(event)
        assert outcome == "Clearance"

    def test_goalkeeper_maps_correctly(self):
        """RED: Goal Keeper should map to 'Goalkeeper'"""
        event = {"type": {"name": "Goal Keeper"}}
        outcome = map_event_to_outcome(event)
        assert outcome == "Goalkeeper"

    def test_other_events_map_to_other(self):
        """RED: Duel, Pressure, etc. should map to 'Other'"""
        test_cases = ["Duel", "Pressure", "Pass", "Shot", "Block", "Interception"]
        for event_type in test_cases:
            event = {"type": {"name": event_type}}
            outcome = map_event_to_outcome(event)
            assert outcome == "Other", f"{event_type} should map to 'Other'"


class TestEventSequenceNavigation:
    """Test finding corners and next events in match event sequences"""

    def test_find_corner_event_by_uuid(self, sample_match_events):
        """RED: Should find corner event by UUID in event list"""
        index = find_event_index(sample_match_events, "corner-uuid-1")
        assert index == 1
        assert sample_match_events[index]["id"] == "corner-uuid-1"

    def test_get_next_event_after_corner(self, sample_match_events):
        """RED: Should return the immediate next event after corner"""
        corner_index = 1  # corner-uuid-1
        next_event = get_next_event(sample_match_events, corner_index)
        assert next_event is not None
        assert next_event["id"] == "next-event-1"
        assert next_event["type"]["name"] == "Ball Receipt*"

    def test_handle_corner_as_last_event(self, sample_match_events):
        """RED: Should handle case where corner is last event in match"""
        # Append corner as last event
        events = sample_match_events + [
            {
                "id": "last-corner",
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Corner"}}
            }
        ]

        last_index = len(events) - 1
        next_event = get_next_event(events, last_index)
        assert next_event is None

    def test_event_not_found_returns_none(self, sample_match_events):
        """RED: Should return None if UUID not found"""
        index = find_event_index(sample_match_events, "nonexistent-uuid")
        assert index is None


class TestLabelExtraction:
    """Test full label extraction pipeline"""

    def test_load_corners_from_file(self, temp_data_dirs, sample_corners):
        """RED: Should load corners_with_freeze_frames.json"""
        # Write sample file
        corners_file = temp_data_dirs["processed"] / "corners_with_freeze_frames.json"
        with open(corners_file, 'w') as f:
            json.dump(sample_corners, f)

        loaded = load_corners(corners_file)
        assert len(loaded) == 2
        assert loaded[0]["match_id"] == "123456"

    def test_load_match_events_file(self, temp_data_dirs, sample_match_events):
        """RED: Should load match events from events directory"""
        # Write sample match file
        match_file = temp_data_dirs["events"] / "123456.json"
        with open(match_file, 'w') as f:
            json.dump(sample_match_events, f)

        loaded = load_match_events(temp_data_dirs["events"], "123456")
        assert len(loaded) == 5
        assert loaded[1]["id"] == "corner-uuid-1"

    def test_add_outcome_to_corner(self, sample_corners, sample_match_events):
        """RED: Should add outcome field to corner based on next event"""
        corner = sample_corners[0]
        labeled = add_outcome_label(corner, sample_match_events)

        assert "outcome" in labeled
        assert labeled["outcome"] == "Ball Receipt"

    def test_handle_missing_next_event(self, sample_corners):
        """RED: Should handle corner with no next event (last in match)"""
        # Events with corner as last event
        events = [
            {"id": "corner-uuid-1", "type": {"name": "Pass"}}
        ]

        corner = sample_corners[0]
        labeled = add_outcome_label(corner, events)

        assert labeled["outcome"] is None or labeled["outcome"] == "Unknown"


class TestOutputValidation:
    """Test output file creation and validation"""

    def test_save_labeled_corners(self, temp_data_dirs, sample_corners):
        """RED: Should save corners with labels to output file"""
        # Add outcomes
        labeled = [
            {**corner, "outcome": "Ball Receipt"}
            for corner in sample_corners
        ]

        output_file = temp_data_dirs["processed"] / "corners_with_labels.json"
        save_labeled_corners(labeled, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert all("outcome" in corner for corner in data)

    def test_validate_class_distribution(self):
        """RED: Should validate outcome class distribution"""
        labeled_corners = [
            {"outcome": "Ball Receipt"} for _ in range(54)
        ] + [
            {"outcome": "Clearance"} for _ in range(23)
        ] + [
            {"outcome": "Goalkeeper"} for _ in range(10)
        ] + [
            {"outcome": "Other"} for _ in range(13)
        ]

        distribution = validate_distribution(labeled_corners)

        assert distribution["Ball Receipt"] == pytest.approx(54.0, abs=1)
        assert distribution["Clearance"] == pytest.approx(23.0, abs=1)
        assert distribution["Goalkeeper"] == pytest.approx(10.0, abs=1)
        assert distribution["Other"] == pytest.approx(13.0, abs=1)
