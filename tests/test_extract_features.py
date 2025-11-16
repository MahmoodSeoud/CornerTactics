"""
Tests for Task 3: Feature Engineering

Tests the extraction of 27 features from corner kick freeze frame data.
"""

import pytest
import numpy as np
from scripts.extract_features import (
    extract_basic_metadata,
    extract_player_counts,
    extract_spatial_density,
    extract_positional_features,
    extract_pass_trajectory,
    extract_all_features
)


class TestBasicMetadata:
    """Test basic corner metadata extraction (5 features)"""

    def test_corner_side_left(self):
        """Test corner side detection for left corner (y < 40)"""
        event = {
            'location': [120.0, 20.0],  # Right side of pitch, y < 40 = left corner
            'period': 1,
            'minute': 15
        }

        features = extract_basic_metadata(event)

        assert 'corner_side' in features
        assert features['corner_side'] == 0  # Left corner

    def test_corner_side_right(self):
        """Test corner side detection for right corner (y >= 40)"""
        event = {
            'location': [120.0, 60.0],  # y >= 40 = right corner
            'period': 1,
            'minute': 15
        }

        features = extract_basic_metadata(event)

        assert features['corner_side'] == 1  # Right corner

    def test_basic_metadata_complete(self):
        """Test all basic metadata fields are extracted"""
        event = {
            'location': [120.0, 80.0],
            'period': 2,
            'minute': 67
        }

        features = extract_basic_metadata(event)

        # Should have exactly 5 features
        assert len(features) == 5
        assert 'corner_side' in features
        assert 'period' in features
        assert 'minute' in features
        assert 'corner_x' in features
        assert 'corner_y' in features

        # Check values
        assert features['corner_x'] == 120.0
        assert features['corner_y'] == 80.0
        assert features['period'] == 2
        assert features['minute'] == 67


class TestPlayerCounts:
    """Test player count features (6 features)"""

    def test_player_counts_in_penalty_box(self):
        """Test counting players in penalty box (x > 102, 18 < y < 62)"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},   # Attacking in box
            {'teammate': True, 'location': [105.0, 30.0]},   # Attacking in box
            {'teammate': False, 'location': [108.0, 35.0]},  # Defending in box
            {'teammate': False, 'location': [106.0, 45.0]},  # Defending in box
            {'teammate': True, 'location': [95.0, 40.0]},    # Attacking outside box
            {'teammate': False, 'location': [100.0, 40.0]}   # Defending outside box
        ]

        features = extract_player_counts(freeze_frame)

        assert features['attacking_in_box'] == 2
        assert features['defending_in_box'] == 2

    def test_player_counts_near_goal(self):
        """Test counting players near goal (x > 108, 30 < y < 50)"""
        freeze_frame = [
            {'teammate': True, 'location': [115.0, 40.0]},   # Attacking near goal
            {'teammate': True, 'location': [110.0, 35.0]},   # Attacking near goal
            {'teammate': False, 'location': [112.0, 40.0]},  # Defending near goal
            {'teammate': True, 'location': [105.0, 40.0]},   # Attacking not near goal
            {'teammate': False, 'location': [108.0, 55.0]}   # Defending not near goal
        ]

        features = extract_player_counts(freeze_frame)

        assert features['attacking_near_goal'] == 2
        assert features['defending_near_goal'] == 1

    def test_player_counts_complete(self):
        """Test all 6 player count features are extracted"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': False, 'location': [108.0, 35.0]}
        ]

        features = extract_player_counts(freeze_frame)

        # Should have exactly 6 features
        assert len(features) == 6
        assert 'total_attacking' in features
        assert 'total_defending' in features
        assert 'attacking_in_box' in features
        assert 'defending_in_box' in features
        assert 'attacking_near_goal' in features
        assert 'defending_near_goal' in features


class TestSpatialDensity:
    """Test spatial density features (4 features)"""

    def test_density_calculation(self):
        """Test density calculation in penalty box"""
        # Penalty box area: (120 - 102) * (62 - 18) = 18 * 44 = 792
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},   # In box
            {'teammate': True, 'location': [105.0, 30.0]},   # In box
            {'teammate': False, 'location': [108.0, 35.0]},  # In box
            {'teammate': False, 'location': [106.0, 45.0]},  # In box
        ]

        features = extract_spatial_density(freeze_frame)

        # 2 attacking / 792 = 0.002525...
        assert abs(features['attacking_density'] - 2/792) < 0.0001
        # 2 defending / 792 = 0.002525...
        assert abs(features['defending_density'] - 2/792) < 0.0001

    def test_numerical_advantage(self):
        """Test numerical advantage calculation"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': True, 'location': [105.0, 30.0]},
            {'teammate': True, 'location': [108.0, 25.0]},
            {'teammate': False, 'location': [108.0, 35.0]},
            {'teammate': False, 'location': [106.0, 45.0]},
        ]

        features = extract_spatial_density(freeze_frame)

        # 3 attacking - 2 defending = 1
        assert features['numerical_advantage'] == 1

    def test_attacker_defender_ratio(self):
        """Test attacker to defender ratio"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': True, 'location': [105.0, 30.0]},
            {'teammate': False, 'location': [108.0, 35.0]},
        ]

        features = extract_spatial_density(freeze_frame)

        # 2 / 1 = 2.0
        assert features['attacker_defender_ratio'] == 2.0

    def test_ratio_with_no_defenders(self):
        """Test ratio when no defenders in box"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': True, 'location': [105.0, 30.0]},
        ]

        features = extract_spatial_density(freeze_frame)

        # Should handle division by zero gracefully (return high value or inf)
        assert features['attacker_defender_ratio'] > 10  # Or np.inf


class TestPositionalFeatures:
    """Test positional features (8 features)"""

    def test_attacking_centroid(self):
        """Test calculation of attacking centroid"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': True, 'location': [106.0, 30.0]},
            {'teammate': False, 'location': [108.0, 35.0]},
        ]

        features = extract_positional_features(freeze_frame)

        # Attacking centroid: ((110 + 106) / 2, (40 + 30) / 2) = (108, 35)
        assert features['attacking_centroid_x'] == 108.0
        assert features['attacking_centroid_y'] == 35.0

    def test_defending_centroid(self):
        """Test calculation of defending centroid"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': False, 'location': [108.0, 36.0]},
            {'teammate': False, 'location': [112.0, 44.0]},
        ]

        features = extract_positional_features(freeze_frame)

        # Defending centroid: ((108 + 112) / 2, (36 + 44) / 2) = (110, 40)
        assert features['defending_centroid_x'] == 110.0
        assert features['defending_centroid_y'] == 40.0

    def test_defending_compactness(self):
        """Test defending line compactness (std of y positions)"""
        freeze_frame = [
            {'teammate': False, 'location': [108.0, 30.0]},
            {'teammate': False, 'location': [108.0, 40.0]},
            {'teammate': False, 'location': [108.0, 50.0]},
        ]

        features = extract_positional_features(freeze_frame)

        # Std of [30, 40, 50] = 10
        expected_std = np.std([30.0, 40.0, 50.0])
        assert abs(features['defending_compactness'] - expected_std) < 0.01

    def test_distance_to_goal(self):
        """Test distance from centroid to goal center (120, 40)"""
        freeze_frame = [
            {'teammate': True, 'location': [110.0, 40.0]},
            {'teammate': True, 'location': [110.0, 40.0]},
        ]

        features = extract_positional_features(freeze_frame)

        # Distance from (110, 40) to (120, 40) = 10
        assert features['attacking_to_goal_dist'] == 10.0


class TestPassTrajectory:
    """Test pass trajectory features (4 features)"""

    def test_pass_end_location(self):
        """Test extraction of pass end location"""
        event = {
            'pass': {
                'end_location': [109.4, 35.5],
                'length': 45.7,
                'height': {'name': 'High Pass'}
            }
        }

        features = extract_pass_trajectory(event)

        assert features['pass_end_x'] == 109.4
        assert features['pass_end_y'] == 35.5

    def test_pass_length(self):
        """Test extraction of pass length"""
        event = {
            'pass': {
                'end_location': [109.4, 35.5],
                'length': 45.7,
                'height': {'name': 'High Pass'}
            }
        }

        features = extract_pass_trajectory(event)

        assert features['pass_length'] == 45.7

    def test_pass_height_encoding(self):
        """Test pass height encoding (Ground=0, Low=1, High=2)"""
        event_ground = {
            'pass': {
                'end_location': [109.4, 35.5],
                'length': 45.7,
                'height': {'name': 'Ground Pass'}
            }
        }

        event_low = {
            'pass': {
                'end_location': [109.4, 35.5],
                'length': 45.7,
                'height': {'name': 'Low Pass'}
            }
        }

        event_high = {
            'pass': {
                'end_location': [109.4, 35.5],
                'length': 45.7,
                'height': {'name': 'High Pass'}
            }
        }

        assert extract_pass_trajectory(event_ground)['pass_height'] == 0
        assert extract_pass_trajectory(event_low)['pass_height'] == 1
        assert extract_pass_trajectory(event_high)['pass_height'] == 2


class TestIntegration:
    """Integration tests for complete feature extraction"""

    def test_extract_all_features_count(self):
        """Test that extract_all_features returns exactly 27 features"""
        corner = {
            'event': {
                'location': [120.0, 80.0],
                'period': 1,
                'minute': 15,
                'pass': {
                    'end_location': [109.4, 35.5],
                    'length': 45.7,
                    'height': {'name': 'High Pass'}
                }
            },
            'freeze_frame': [
                {'teammate': True, 'location': [110.0, 40.0]},
                {'teammate': False, 'location': [108.0, 35.0]}
            ]
        }

        features = extract_all_features(corner)

        # Should have exactly 27 features
        assert len(features) == 27

    def test_extract_all_features_types(self):
        """Test that all features are numeric"""
        corner = {
            'event': {
                'location': [120.0, 80.0],
                'period': 1,
                'minute': 15,
                'pass': {
                    'end_location': [109.4, 35.5],
                    'length': 45.7,
                    'height': {'name': 'High Pass'}
                }
            },
            'freeze_frame': [
                {'teammate': True, 'location': [110.0, 40.0]},
                {'teammate': False, 'location': [108.0, 35.0]}
            ]
        }

        features = extract_all_features(corner)

        # All values should be numeric (int or float)
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"Feature {key} has non-numeric value: {value}"
