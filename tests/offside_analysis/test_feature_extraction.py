"""Tests for offside-predictive feature extraction.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_freeze_frame():
    """Sample freeze frame with known positions for testing.

    Pitch: 120x80, goal at x=120
    Setup: 5 attackers, 5 defenders including goalkeeper
    """
    return [
        # Attackers (teammate=True) - various positions
        {"location": [108.0, 30.0], "teammate": True, "keeper": False, "actor": False},
        {"location": [110.0, 40.0], "teammate": True, "keeper": False, "actor": False},
        {"location": [112.0, 35.0], "teammate": True, "keeper": False, "actor": False},  # Beyond last defender
        {"location": [105.0, 45.0], "teammate": True, "keeper": False, "actor": False},
        {"location": [120.0, 0.0], "teammate": True, "keeper": False, "actor": True},  # Corner taker
        # Defenders (teammate=False) - defensive line at ~110
        {"location": [110.0, 32.0], "teammate": False, "keeper": False, "actor": False},
        {"location": [110.0, 38.0], "teammate": False, "keeper": False, "actor": False},
        {"location": [110.0, 44.0], "teammate": False, "keeper": False, "actor": False},
        {"location": [108.0, 50.0], "teammate": False, "keeper": False, "actor": False},
        {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},  # Goalkeeper
    ]


@pytest.fixture
def sample_corner(sample_freeze_frame):
    """Complete corner data structure."""
    return {
        "match_id": "12345",
        "event": {
            "id": "test-corner",
            "location": [120.0, 0.0],  # Corner from right side
        },
        "freeze_frame": sample_freeze_frame,
        "shot_outcome": 1,
    }


class TestDefensiveLineFeatures:
    """Test extraction of defensive line features."""

    def test_find_last_defender_x(self, sample_freeze_frame):
        """Should find x-position of last defender (excluding goalkeeper)."""
        from experiments.offside_analysis.feature_extraction import find_last_defender_x

        # Last defender (excluding GK) is at x=110
        last_def_x = find_last_defender_x(sample_freeze_frame)

        assert last_def_x == 110.0

    def test_last_defender_excludes_goalkeeper(self, sample_freeze_frame):
        """Last defender calculation should exclude goalkeeper."""
        from experiments.offside_analysis.feature_extraction import find_last_defender_x

        # Goalkeeper is at x=120, but should be excluded
        last_def_x = find_last_defender_x(sample_freeze_frame)

        assert last_def_x < 120.0

    def test_defensive_line_y_spread(self, sample_freeze_frame):
        """Should compute y-axis spread of defensive line."""
        from experiments.offside_analysis.feature_extraction import compute_defensive_line_spread

        # Defenders at y=32, 38, 44, 50 (excluding GK at 40)
        spread = compute_defensive_line_spread(sample_freeze_frame)

        # Spread = max(y) - min(y) = 50 - 32 = 18
        assert spread == pytest.approx(18.0, abs=0.1)

    def test_defensive_line_compactness(self, sample_freeze_frame):
        """Should compute compactness (std dev of x-positions)."""
        from experiments.offside_analysis.feature_extraction import compute_defensive_compactness

        compactness = compute_defensive_compactness(sample_freeze_frame)

        # Should be low since defenders are mostly at x=110
        assert compactness < 5.0  # Low spread


class TestAttackerPositionFeatures:
    """Test extraction of attacker position features."""

    def test_count_attackers_beyond_defender(self, sample_freeze_frame):
        """Should count attackers beyond the last defender."""
        from experiments.offside_analysis.feature_extraction import count_attackers_beyond_defender

        # Last defender at x=110, attacker at x=112 is beyond
        count = count_attackers_beyond_defender(sample_freeze_frame)

        assert count >= 1  # At least the one at 112

    def test_furthest_attacker_x(self, sample_freeze_frame):
        """Should find x-position of furthest forward attacker."""
        from experiments.offside_analysis.feature_extraction import find_furthest_attacker_x

        # Furthest non-corner-taker attacker is at x=112
        furthest_x = find_furthest_attacker_x(sample_freeze_frame, exclude_corner_taker=True)

        assert furthest_x == 112.0

    def test_attacker_to_defender_gap(self, sample_freeze_frame):
        """Should compute gap between furthest attacker and last defender."""
        from experiments.offside_analysis.feature_extraction import compute_attacker_defender_gap

        # Furthest attacker at 112, last defender at 110 -> gap = 2
        gap = compute_attacker_defender_gap(sample_freeze_frame)

        assert gap == pytest.approx(2.0, abs=0.1)

    def test_attackers_in_offside_zone(self, sample_freeze_frame):
        """Should count attackers in potential offside zone."""
        from experiments.offside_analysis.feature_extraction import count_attackers_in_offside_zone

        # Offside zone: x > last_defender_x and x < 120 (goal line)
        count = count_attackers_in_offside_zone(sample_freeze_frame)

        assert count >= 1


class TestExtractAllFeatures:
    """Test comprehensive feature extraction."""

    def test_extract_offside_features_returns_dict(self, sample_corner):
        """Should return dictionary of all offside-related features."""
        from experiments.offside_analysis.feature_extraction import extract_offside_features

        features = extract_offside_features(sample_corner)

        assert isinstance(features, dict)
        assert 'last_defender_x' in features
        assert 'attackers_beyond_defender' in features
        assert 'attacker_defender_gap' in features

    def test_extract_offside_features_values(self, sample_corner):
        """Feature values should be reasonable."""
        from experiments.offside_analysis.feature_extraction import extract_offside_features

        features = extract_offside_features(sample_corner)

        # All features should be numeric
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"

        # Position features should be in pitch bounds
        assert 0 <= features['last_defender_x'] <= 120
        assert features['attackers_beyond_defender'] >= 0


class TestBatchExtraction:
    """Test batch feature extraction from multiple corners."""

    def test_extract_features_batch(self, sample_corner):
        """Should extract features from list of corners."""
        from experiments.offside_analysis.feature_extraction import extract_features_batch

        corners = [sample_corner, sample_corner, sample_corner]
        features_df = extract_features_batch(corners)

        assert len(features_df) == 3
        assert 'last_defender_x' in features_df.columns

    def test_batch_extraction_handles_missing_data(self):
        """Should handle corners with incomplete data."""
        from experiments.offside_analysis.feature_extraction import extract_features_batch

        corners = [
            {"match_id": "1", "event": {"id": "c1", "location": [120.0, 0.0]},
             "freeze_frame": [], "shot_outcome": 0},  # Empty freeze frame
        ]

        # Should not raise, but may have NaN values
        features_df = extract_features_batch(corners)
        assert len(features_df) == 1


class TestEdgeCases:
    """Test edge cases in feature extraction."""

    def test_no_defenders(self):
        """Should handle case with no defenders visible."""
        from experiments.offside_analysis.feature_extraction import find_last_defender_x

        freeze_frame = [
            {"location": [110.0, 40.0], "teammate": True, "keeper": False, "actor": False},
        ]

        last_def_x = find_last_defender_x(freeze_frame)

        # Should return None or default value
        assert last_def_x is None or last_def_x == 120.0

    def test_only_goalkeeper(self):
        """Should handle case with only goalkeeper as defender."""
        from experiments.offside_analysis.feature_extraction import find_last_defender_x

        freeze_frame = [
            {"location": [110.0, 40.0], "teammate": True, "keeper": False, "actor": False},
            {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},
        ]

        last_def_x = find_last_defender_x(freeze_frame)

        # No outfield defenders, should return None or goal line
        assert last_def_x is None or last_def_x == 120.0

    def test_corner_taker_exclusion(self):
        """Corner taker should be excluded from attacker analysis."""
        from experiments.offside_analysis.feature_extraction import find_furthest_attacker_x

        freeze_frame = [
            {"location": [120.0, 0.0], "teammate": True, "keeper": False, "actor": True},  # Corner taker
            {"location": [105.0, 40.0], "teammate": True, "keeper": False, "actor": False},
        ]

        furthest_x = find_furthest_attacker_x(freeze_frame, exclude_corner_taker=True)

        # Should be 105, not 120 (corner taker)
        assert furthest_x == 105.0
