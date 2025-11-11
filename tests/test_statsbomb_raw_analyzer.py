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
        assert hasattr(analyzer, 'sb')  # Has StatsBomb SDK
        assert analyzer.events == []
        assert analyzer.competitions == []

    def test_fetch_competitions(self):
        """Test fetching competitions from SDK."""
        analyzer = StatsBombRawAnalyzer()

        # Mock the SDK's competitions method
        mock_comps = pd.DataFrame([
            {"competition_id": 16, "season_id": 4, "competition_name": "Champions League"},
            {"competition_id": 11, "season_id": 1, "competition_name": "La Liga"}
        ])

        with patch.object(analyzer.sb, 'competitions', return_value=mock_comps):
            competitions = analyzer.fetch_competitions()

            assert len(competitions) == 2
            assert competitions.iloc[0]["competition_name"] == "Champions League"

    def test_fetch_match_events(self):
        """Test fetching events for a specific match."""
        analyzer = StatsBombRawAnalyzer()

        # Mock the SDK's events method
        mock_events = pd.DataFrame([
            {
                "id": "event1",
                "type": "Pass",
                "team": "Barcelona",
                "player": "Messi",
                "location": [60.0, 40.0],
                "timestamp": "00:00:15.123"
            },
            {
                "id": "event2",
                "type": "Ball Receipt*",
                "team": "Barcelona",
                "location": [65.0, 42.0],
                "timestamp": "00:00:16.456"
            }
        ])

        with patch.object(analyzer.sb, 'events', return_value=mock_events):
            events = analyzer.fetch_match_events(12345)

            assert len(events) == 2
            assert events.iloc[0]["type"] == "Pass"
            assert events.iloc[1]["type"] == "Ball Receipt*"

    def test_identify_corner_kicks(self):
        """Test identification of corner kicks in events (SDK format)."""
        events = [
            {
                "type": "Pass",  # SDK flattens
                "pass_type": "Corner",  # SDK uses pass_type field
                "team": "Barcelona"
            },
            {
                "type": "Pass",
                "pass_type": "Normal",
                "team": "Barcelona"
            },
            {
                "type": "Shot",
                "team": "Barcelona"
            },
            {
                "type": "Pass",
                "pass_type": "Corner",
                "team": "Real Madrid"
            }
        ]

        analyzer = StatsBombRawAnalyzer()
        analyzer.events = events
        corners = analyzer.identify_corner_kicks()

        assert len(corners) == 2
        assert corners[0]["team"] == "Barcelona"
        assert corners[1]["team"] == "Real Madrid"


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
        """Test tracking corner-specific transitions (SDK format)."""
        events = [
            {"type": "Pass", "pass_type": "Corner"},  # SDK format
            {"type": "Ball Receipt*"},
            {"type": "Shot"},
            {"type": "Pass", "pass_type": "Corner"},
            {"type": "Clearance"},
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
        """Test extracting features from a single event (SDK format)."""
        event = {
            "id": "abc123",
            "type": "Pass",  # SDK flattens this
            "team": "Barcelona",
            "player": "Messi",
            "location": [60.0, 40.0],
            "timestamp": "00:15:30.123",
            "under_pressure": True,
            "pass_length": 25.5,  # SDK flattens to pass_*
            "pass_angle": 1.57,
            "pass_height": "High Pass",
            "pass_end_location": [85.0, 45.0]
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
        """Test summarizing features across all events (SDK format)."""
        events = [
            {
                "type": "Pass",  # SDK flattens
                "location": [60.0, 40.0],
                "under_pressure": True,
                "pass_length": 20.0  # SDK uses pass_*
            },
            {
                "type": "Shot",
                "location": [100.0, 40.0],
                "shot_statsbomb_xg": 0.25  # SDK uses shot_*
            },
            {
                "type": "Clearance",
                "location": [30.0, 40.0],
                "clearance_aerial_won": True  # SDK uses clearance_*
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