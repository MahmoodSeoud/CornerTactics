#!/usr/bin/env python3
"""
Integration tests for StatsBomb raw data analysis.
"""

import pytest
import json
import pandas as pd
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statsbomb_raw_analyzer import StatsBombRawAnalyzer


class TestStatsBombIntegration:
    """Integration tests for the complete analysis pipeline."""

    @patch('requests.get')
    def test_complete_analysis_pipeline(self, mock_get):
        """Test the complete analysis pipeline with mock data."""
        # Mock competition response (SDK returns DataFrame)
        mock_comps_df = pd.DataFrame([
            {
                "competition_id": 16,
                "season_id": 4,
                "competition_name": "Champions League",
                "season_name": "2019/2020"
            }
        ])

        # Mock matches response (SDK returns DataFrame)
        mock_matches_df = pd.DataFrame([
            {
                "match_id": 12345,
                "home_team": "Barcelona",
                "away_team": "Liverpool"
            }
        ])

        # Mock events with corner kick and following events (SDK format)
        mock_events_df = pd.DataFrame([
            {
                "id": "1",
                "type": "Pass",  # SDK flattens
                "pass_type": "Corner",  # SDK uses pass_type
                "pass_end_location": [110, 40],
                "team": "Barcelona",
                "player": "Messi",
                "location": [120, 0],
                "timestamp": "00:10:00.000"
            },
            {
                "id": "2",
                "type": "Ball Receipt*",
                "team": "Barcelona",
                "player": "Pique",
                "location": [110, 40],
                "timestamp": "00:10:01.000"
            },
            {
                "id": "3",
                "type": "Shot",
                "shot_outcome": "Saved",  # SDK flattens
                "shot_statsbomb_xg": 0.15,
                "team": "Barcelona",
                "player": "Pique",
                "location": [108, 40],
                "timestamp": "00:10:02.000"
            },
            {
                "id": "4",
                "type": "Goal Keeper",
                "team": "Liverpool",
                "player": "Alisson",
                "location": [120, 40],
                "timestamp": "00:10:02.500"
            }
        ])

        # Mock SDK methods
        analyzer = StatsBombRawAnalyzer()

        with patch.object(analyzer.sb, 'competitions', return_value=mock_comps_df):
            with patch.object(analyzer.sb, 'matches', return_value=mock_matches_df):
                with patch.object(analyzer.sb, 'events', return_value=mock_events_df):
                    # Run analysis
                    results = analyzer.analyze(num_matches=1)

                    # Verify results
                    assert results['num_events'] == 4
                    assert results['num_corners'] == 1
                    assert results['num_matches'] == 1

                    # Check transition matrix
                    matrix = results['transition_matrix']
                    assert 'Corner' in matrix.index
                    assert matrix.loc['Corner', 'Ball Receipt*'] > 0

                    # Check features
                    features = results['feature_summary']
                    assert 'Pass' in features['event_types']
                    assert 'Shot' in features['event_types']
                    assert features['location_coverage'] == 1.0  # All events have location

                    # Check corner sequences (SDK format)
                    sequences = results['corner_sequences']
                    assert len(sequences) == 1
                    assert sequences[0]['corner']['player'] == 'Messi'  # SDK flattens
                    assert len(sequences[0]['following_events']) == 3

    def test_transition_probability_calculation(self):
        """Test that transition probabilities sum to 1."""
        from src.statsbomb_raw_analyzer import TransitionMatrixBuilder

        builder = TransitionMatrixBuilder()

        # Add transitions
        builder.add_transition("Corner", "Ball Receipt*")
        builder.add_transition("Corner", "Ball Receipt*")
        builder.add_transition("Corner", "Clearance")
        builder.add_transition("Ball Receipt*", "Shot")
        builder.add_transition("Ball Receipt*", "Pass")

        matrix = builder.calculate_probability_matrix()

        # Check that each row sums to 1 (or 0 if no transitions)
        for idx in matrix.index:
            row_sum = matrix.loc[idx].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.001, f"Row {idx} sums to {row_sum}, not 1"

    def test_corner_specific_analysis(self):
        """Test corner-specific transition tracking (SDK format)."""
        from src.statsbomb_raw_analyzer import TransitionMatrixBuilder

        events = [
            {"type": "Pass", "pass_type": "Corner"},  # SDK format
            {"type": "Ball Receipt*"},
            {"type": "Shot"},
            {"type": "Pass", "pass_type": "Normal"},  # Regular pass
            {"type": "Ball Receipt*"},
            {"type": "Pass", "pass_type": "Corner"},
            {"type": "Clearance"}
        ]

        builder = TransitionMatrixBuilder()
        builder.build_from_events(events, track_corners=True)

        # Get corner transitions
        corner_trans = builder.get_corner_transitions()

        assert len(corner_trans) == 2  # Ball Receipt* and Clearance
        assert corner_trans['Ball Receipt*'] == 0.5
        assert corner_trans['Clearance'] == 0.5

    def test_feature_extraction_completeness(self):
        """Test that all feature types are properly extracted (SDK format)."""
        from src.statsbomb_raw_analyzer import FeatureExtractor

        events = [
            {
                "type": "Pass",  # SDK flattens
                "location": [60, 40],
                "timestamp": "00:00:15",
                "under_pressure": True,
                "pass_length": 25.5,  # SDK uses pass_*
                "pass_angle": 1.57,
                "pass_height": "High Pass"
            },
            {
                "type": "Shot",
                "location": [100, 40],
                "timestamp": "00:00:20",
                "shot_statsbomb_xg": 0.25,  # SDK uses shot_*
                "shot_outcome": "Goal",
                "shot_technique": "Normal"
            },
            {
                "type": "Clearance",
                "location": [30, 40],
                "timestamp": "00:00:25",
                "clearance_aerial_won": True,  # SDK uses clearance_*
                "clearance_head": True
            }
        ]

        extractor = FeatureExtractor()
        summary = extractor.summarize_features(events)

        # Check all event types captured
        assert len(summary['event_types']) == 3
        assert set(summary['event_types']) == {"Pass", "Shot", "Clearance"}

        # Check coverage calculations
        assert summary['location_coverage'] == 1.0
        assert summary['timestamp_coverage'] == 1.0
        assert summary['pressure_rate'] == 1/3

        # Check type-specific features (SDK uses pass_*, shot_*, etc.)
        assert 'pass_length' in summary['type_specific_features']['pass']
        assert 'shot_statsbomb_xg' in summary['type_specific_features']['shot']
        assert 'clearance_aerial_won' in summary['type_specific_features']['clearance']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])