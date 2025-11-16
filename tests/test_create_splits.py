"""
Tests for Task 4: Train/Test Split Creation

Tests the creation of match-based stratified train/test splits.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def features_csv_path():
    """Path to the features CSV file"""
    return Path("data/processed/corners_with_features.csv")


@pytest.fixture
def train_indices_path():
    """Path to the train indices CSV file"""
    return Path("data/processed/train_indices.csv")


@pytest.fixture
def test_indices_path():
    """Path to the test indices CSV file"""
    return Path("data/processed/test_indices.csv")


class TestLoadFeaturesCSV:
    """Test loading features CSV file"""

    def test_features_csv_exists(self, features_csv_path):
        """Test that features CSV file exists"""
        assert features_csv_path.exists(), f"Features CSV not found at {features_csv_path}"

    def test_features_csv_has_required_columns(self, features_csv_path):
        """Test that CSV has required columns"""
        df = pd.read_csv(features_csv_path)

        required_columns = ['match_id', 'event_id', 'outcome']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_features_csv_has_correct_row_count(self, features_csv_path):
        """Test that CSV has expected number of corners (~1,933)"""
        df = pd.read_csv(features_csv_path)

        # Allow some tolerance
        assert 1900 <= len(df) <= 1950, f"Expected ~1,933 corners, got {len(df)}"

    def test_features_csv_has_no_missing_match_ids(self, features_csv_path):
        """Test that all rows have valid match_ids"""
        df = pd.read_csv(features_csv_path)

        assert df['match_id'].notna().all(), "Some rows have missing match_id"

    def test_features_csv_has_no_missing_outcomes(self, features_csv_path):
        """Test that all rows have valid outcomes"""
        df = pd.read_csv(features_csv_path)

        assert df['outcome'].notna().all(), "Some rows have missing outcome"


class TestMatchBasedSplit:
    """Test match-based splitting logic"""

    def test_split_files_exist(self, train_indices_path, test_indices_path):
        """Test that split index files are created"""
        # This test will fail initially - we need to implement the script
        assert train_indices_path.exists(), f"Train indices not found at {train_indices_path}"
        assert test_indices_path.exists(), f"Test indices not found at {test_indices_path}"

    def test_no_match_overlap_between_train_and_test(self, features_csv_path, train_indices_path, test_indices_path):
        """Test that no match appears in both train and test sets"""
        df = pd.read_csv(features_csv_path)
        train_indices = pd.read_csv(train_indices_path)
        test_indices = pd.read_csv(test_indices_path)

        train_matches = set(df.iloc[train_indices['index']]['match_id'].unique())
        test_matches = set(df.iloc[test_indices['index']]['match_id'].unique())

        overlap = train_matches & test_matches
        assert len(overlap) == 0, f"Found {len(overlap)} matches in both train and test"

    def test_all_corners_in_either_train_or_test(self, features_csv_path, train_indices_path, test_indices_path):
        """Test that all corners are assigned to either train or test"""
        df = pd.read_csv(features_csv_path)
        train_indices = pd.read_csv(train_indices_path)
        test_indices = pd.read_csv(test_indices_path)

        total_in_splits = len(train_indices) + len(test_indices)
        assert total_in_splits == len(df), f"Expected {len(df)} corners in splits, got {total_in_splits}"

    def test_split_ratio_approximately_80_20(self, train_indices_path, test_indices_path):
        """Test that split ratio is approximately 80/20"""
        train_indices = pd.read_csv(train_indices_path)
        test_indices = pd.read_csv(test_indices_path)

        total = len(train_indices) + len(test_indices)
        train_ratio = len(train_indices) / total

        # Allow ±5% tolerance due to match-based grouping
        assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio:.2%} not within 75-85%"


class TestStratification:
    """Test stratification of outcome classes"""

    def test_train_class_distribution_matches_overall(self, features_csv_path, train_indices_path):
        """Test that train set has similar class distribution to overall"""
        df = pd.read_csv(features_csv_path)
        train_indices = pd.read_csv(train_indices_path)

        overall_dist = df['outcome'].value_counts(normalize=True)
        train_dist = df.iloc[train_indices['index']]['outcome'].value_counts(normalize=True)

        # Check each class is within ±5% of overall distribution
        for outcome_class in overall_dist.index:
            overall_pct = overall_dist[outcome_class]
            train_pct = train_dist.get(outcome_class, 0)

            diff = abs(overall_pct - train_pct)
            assert diff <= 0.05, f"{outcome_class}: train {train_pct:.2%} differs from overall {overall_pct:.2%} by {diff:.2%}"

    def test_test_class_distribution_matches_overall(self, features_csv_path, test_indices_path):
        """Test that test set has similar class distribution to overall"""
        df = pd.read_csv(features_csv_path)
        test_indices = pd.read_csv(test_indices_path)

        overall_dist = df['outcome'].value_counts(normalize=True)
        test_dist = df.iloc[test_indices['index']]['outcome'].value_counts(normalize=True)

        # Check each class is within ±5% of overall distribution
        for outcome_class in overall_dist.index:
            overall_pct = overall_dist[outcome_class]
            test_pct = test_dist.get(outcome_class, 0)

            diff = abs(overall_pct - test_pct)
            assert diff <= 0.05, f"{outcome_class}: test {test_pct:.2%} differs from overall {overall_pct:.2%} by {diff:.2%}"

    def test_all_classes_present_in_train(self, features_csv_path, train_indices_path):
        """Test that all outcome classes are present in train set"""
        df = pd.read_csv(features_csv_path)
        train_indices = pd.read_csv(train_indices_path)

        overall_classes = set(df['outcome'].unique())
        train_classes = set(df.iloc[train_indices['index']]['outcome'].unique())

        missing_classes = overall_classes - train_classes
        assert len(missing_classes) == 0, f"Missing classes in train: {missing_classes}"

    def test_all_classes_present_in_test(self, features_csv_path, test_indices_path):
        """Test that all outcome classes are present in test set"""
        df = pd.read_csv(features_csv_path)
        test_indices = pd.read_csv(test_indices_path)

        overall_classes = set(df['outcome'].unique())
        test_classes = set(df.iloc[test_indices['index']]['outcome'].unique())

        missing_classes = overall_classes - test_classes
        assert len(missing_classes) == 0, f"Missing classes in test: {missing_classes}"


class TestOutputFormat:
    """Test output CSV format"""

    def test_train_indices_has_index_column(self, train_indices_path):
        """Test that train indices CSV has 'index' column"""
        train_indices = pd.read_csv(train_indices_path)
        assert 'index' in train_indices.columns, "Train indices missing 'index' column"

    def test_test_indices_has_index_column(self, test_indices_path):
        """Test that test indices CSV has 'index' column"""
        test_indices = pd.read_csv(test_indices_path)
        assert 'index' in test_indices.columns, "Test indices missing 'index' column"

    def test_train_indices_are_valid(self, features_csv_path, train_indices_path):
        """Test that train indices are valid row numbers"""
        df = pd.read_csv(features_csv_path)
        train_indices = pd.read_csv(train_indices_path)

        max_index = train_indices['index'].max()
        assert max_index < len(df), f"Train index {max_index} out of bounds for {len(df)} rows"

        min_index = train_indices['index'].min()
        assert min_index >= 0, f"Train index {min_index} is negative"

    def test_test_indices_are_valid(self, features_csv_path, test_indices_path):
        """Test that test indices are valid row numbers"""
        df = pd.read_csv(features_csv_path)
        test_indices = pd.read_csv(test_indices_path)

        max_index = test_indices['index'].max()
        assert max_index < len(df), f"Test index {max_index} out of bounds for {len(df)} rows"

        min_index = test_indices['index'].min()
        assert min_index >= 0, f"Test index {min_index} is negative"
