#!/usr/bin/env python3
"""
Tests for StatsBomb raw data analyzer.
"""

import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statsbomb_raw_analyzer import (
    StatsBombRawAnalyzer,
    TransitionMatrixBuilder,
    FeatureExtractor,
    ReportGenerator
)


class TestStatsBombRawAnalyzer:
    """Test the main StatsBomb raw data analyzer."""

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        analyzer = StatsBombRawAnalyzer()
        assert analyzer is not None
        assert analyzer.base_url == "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
        assert analyzer.events == []
        assert analyzer.competitions == []

    @patch('requests.get')
    def test_fetch_competitions(self, mock_get):
        """Test fetching competitions from GitHub."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"competition_id": 16, "season_id": 4, "competition_name": "Champions League"},
            {"competition_id": 11, "season_id": 1, "competition_name": "La Liga"}
        ]
        mock_get.return_value = mock_response

        analyzer = StatsBombRawAnalyzer()
        competitions = analyzer.fetch_competitions()

        assert len(competitions) == 2
        assert competitions[0]["competition_name"] == "Champions League"
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_fetch_match_events(self, mock_get):
        """Test fetching events for a specific match."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "event1",
                "type": {"name": "Pass"},
                "team": {"name": "Barcelona"},
                "player": {"name": "Messi"},
                "location": [60.0, 40.0],
                "timestamp": "00:00:15.123"
            },
            {
                "id": "event2",
                "type": {"name": "Ball Receipt*"},
                "team": {"name": "Barcelona"},
                "location": [65.0, 42.0],
                "timestamp": "00:00:16.456"
            }
        ]
        mock_get.return_value = mock_response

        analyzer = StatsBombRawAnalyzer()
        events = analyzer.fetch_match_events(12345)

        assert len(events) == 2
        assert events[0]["type"]["name"] == "Pass"
        assert events[1]["type"]["name"] == "Ball Receipt*"

    def test_identify_corner_kicks(self):
        """Test identification of corner kicks in events."""
        events = [
            {
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Corner"}},
                "team": {"name": "Barcelona"}
            },
            {
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Normal"}},
                "team": {"name": "Barcelona"}
            },
            {
                "type": {"name": "Shot"},
                "team": {"name": "Barcelona"}
            },
            {
                "type": {"name": "Pass"},
                "pass": {"type": {"name": "Corner"}},
                "team": {"name": "Real Madrid"}
            }
        ]

        analyzer = StatsBombRawAnalyzer()
        analyzer.events = events
        corners = analyzer.identify_corner_kicks()

        assert len(corners) == 2
        assert corners[0]["team"]["name"] == "Barcelona"
        assert corners[1]["team"]["name"] == "Real Madrid"


class TestTransitionMatrixBuilder:
    """Test transition matrix builder."""

    def test_builder_initialization(self):
        """Test transition matrix builder initialization."""
        builder = TransitionMatrixBuilder()
        assert builder is not None
        assert builder.transitions == {}
        assert builder.event_counts == {}

    def test_add_transition(self):
        """Test adding a transition."""
        builder = TransitionMatrixBuilder()
        builder.add_transition("Pass", "Ball Receipt*")
        builder.add_transition("Pass", "Ball Receipt*")
        builder.add_transition("Pass", "Interception")

        assert builder.transitions["Pass"]["Ball Receipt*"] == 2
        assert builder.transitions["Pass"]["Interception"] == 1

    def test_calculate_probabilities(self):
        """Test probability calculation."""
        builder = TransitionMatrixBuilder()
        builder.add_transition("Corner", "Ball Receipt*")
        builder.add_transition("Corner", "Ball Receipt*")
        builder.add_transition("Corner", "Clearance")
        builder.add_transition("Corner", "Clearance")
        builder.add_transition("Corner", "Goal Keeper")

        matrix = builder.calculate_probability_matrix()

        assert matrix.loc["Corner", "Ball Receipt*"] == pytest.approx(0.4)
        assert matrix.loc["Corner", "Clearance"] == pytest.approx(0.4)
        assert matrix.loc["Corner", "Goal Keeper"] == pytest.approx(0.2)

    def test_build_from_events(self):
        """Test building transition matrix from event sequence."""
        events = [
            {"type": {"name": "Pass"}},
            {"type": {"name": "Ball Receipt*"}},
            {"type": {"name": "Carry"}},
            {"type": {"name": "Pass"}},
            {"type": {"name": "Ball Receipt*"}},
            {"type": {"name": "Shot"}},
            {"type": {"name": "Goal Keeper"}}
        ]

        builder = TransitionMatrixBuilder()
        builder.build_from_events(events)

        assert builder.transitions["Pass"]["Ball Receipt*"] == 2
        assert builder.transitions["Ball Receipt*"]["Carry"] == 1
        assert builder.transitions["Ball Receipt*"]["Shot"] == 1
        assert builder.transitions["Shot"]["Goal Keeper"] == 1

    def test_corner_specific_transitions(self):
        """Test tracking corner-specific transitions."""
        events = [
            {"type": {"name": "Pass"}, "pass": {"type": {"name": "Corner"}}},
            {"type": {"name": "Ball Receipt*"}},
            {"type": {"name": "Shot"}},
            {"type": {"name": "Pass"}, "pass": {"type": {"name": "Corner"}}},
            {"type": {"name": "Clearance"}},
        ]

        builder = TransitionMatrixBuilder()
        builder.build_from_events(events, track_corners=True)

        assert "Corner" in builder.transitions
        assert builder.transitions["Corner"]["Ball Receipt*"] == 1
        assert builder.transitions["Corner"]["Clearance"] == 1


class TestFeatureExtractor:
    """Test feature extraction from raw data."""

    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert extractor.features == {}

    def test_extract_event_features(self):
        """Test extracting features from a single event."""
        event = {
            "id": "abc123",
            "type": {"name": "Pass"},
            "team": {"name": "Barcelona"},
            "player": {"name": "Messi"},
            "location": [60.0, 40.0],
            "timestamp": "00:15:30.123",
            "under_pressure": True,
            "pass": {
                "length": 25.5,
                "angle": 1.57,
                "height": {"name": "High Pass"},
                "end_location": [85.0, 45.0]
            }
        }

        extractor = FeatureExtractor()
        features = extractor.extract_event_features(event)

        assert features["has_location"] == True
        assert features["has_timestamp"] == True
        assert features["has_under_pressure"] == True
        assert features["event_type"] == "Pass"
        assert features["pass_features"]["length"] == 25.5
        assert features["pass_features"]["has_end_location"] == True

    def test_summarize_all_features(self):
        """Test summarizing features across all events."""
        events = [
            {
                "type": {"name": "Pass"},
                "location": [60.0, 40.0],
                "under_pressure": True,
                "pass": {"length": 20.0}
            },
            {
                "type": {"name": "Shot"},
                "location": [100.0, 40.0],
                "shot": {"statsbomb_xg": 0.25}
            },
            {
                "type": {"name": "Clearance"},
                "location": [30.0, 40.0],
                "clearance": {"aerial_won": True}
            }
        ]

        extractor = FeatureExtractor()
        summary = extractor.summarize_features(events)

        assert set(summary["event_types"]) == {"Pass", "Shot", "Clearance"}
        assert summary["location_coverage"] == 1.0  # All events have location
        assert summary["pressure_rate"] == pytest.approx(1/3)
        assert "pass" in summary["type_specific_features"]
        assert "shot" in summary["type_specific_features"]


class TestReportGenerator:
    """Test report generation."""

    def test_generator_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator()
        assert generator is not None

    def test_generate_transition_report(self):
        """Test generating transition probability report."""
        matrix = pd.DataFrame({
            "Ball Receipt*": [0.6, 0.6, 0.0],
            "Clearance": [0.3, 0.3, 0.0],
            "Shot": [0.1, 0.1, 0.0]
        }, index=["Corner", "Ball Receipt*", "Shot"])

        generator = ReportGenerator()
        report = generator.generate_transition_report(matrix, "Corner")

        assert "Corner" in report
        assert "Ball Receipt*: 0.600" in report
        assert "Clearance: 0.300" in report
        assert "Shot: 0.100" in report

    def test_generate_feature_report(self):
        """Test generating feature documentation report."""
        feature_summary = {
            "event_types": {"Pass", "Shot", "Clearance"},
            "location_coverage": 0.95,
            "timestamp_coverage": 1.0,
            "type_specific_features": {
                "pass": ["length", "angle", "height"],
                "shot": ["statsbomb_xg", "outcome"]
            }
        }

        generator = ReportGenerator()
        report = generator.generate_feature_report(feature_summary)

        assert "Event Types" in report
        assert "Pass" in report
        assert "Location Coverage: 95.0%" in report
        assert "pass features" in report.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])