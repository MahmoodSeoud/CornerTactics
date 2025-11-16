"""
Tests for Task 8: Binary Shot Classification Models
"""
import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


class TestDataLoading:
    """Test data loading and merging of shot labels with features"""

    def test_load_shot_labels(self):
        """Test loading shot labels from JSON"""
        shot_labels_path = Path("data/processed/corners_with_shot_labels.json")
        assert shot_labels_path.exists(), "Shot labels file should exist"

        with open(shot_labels_path) as f:
            data = json.load(f)

        assert len(data) == 1933, "Should have 1933 corners"
        assert all('shot_outcome' in corner for corner in data), "All corners should have shot_outcome"

        # Check binary labels
        shot_outcomes = [corner['shot_outcome'] for corner in data]
        assert all(outcome in [0, 1] for outcome in shot_outcomes), "Labels should be binary (0 or 1)"

    def test_load_features(self):
        """Test loading feature data"""
        features_path = Path("data/processed/corners_with_features.csv")
        assert features_path.exists(), "Features file should exist"

        df = pd.read_csv(features_path)
        assert len(df) == 1933, "Should have 1933 corners"
        assert 'event_id' in df.columns, "Should have event_id column"
        assert 'match_id' in df.columns, "Should have match_id column"

    def test_merge_shot_labels_with_features(self):
        """Test merging shot labels with feature data"""
        # This will be implemented in the script
        # For now, just test that we can load both datasets

        # Load shot labels
        with open("data/processed/corners_with_shot_labels.json") as f:
            shot_data = json.load(f)

        # Load features
        features_df = pd.read_csv("data/processed/corners_with_features.csv")

        # Create shot labels DataFrame
        shot_labels = []
        for corner in shot_data:
            event_id = corner['event']['id']
            shot_outcome = corner['shot_outcome']
            shot_labels.append({'event_id': event_id, 'shot_outcome': shot_outcome})

        shot_df = pd.DataFrame(shot_labels)

        # Merge
        merged = pd.merge(features_df, shot_df, on='event_id', how='inner')

        assert len(merged) == 1933, "Should have all 1933 corners after merge"
        assert 'shot_outcome' in merged.columns, "Should have shot_outcome column"
        assert merged['shot_outcome'].isin([0, 1]).all(), "Shot outcomes should be binary"


class TestRandomForestBinary:
    """Test Random Forest binary classifier"""

    def test_random_forest_training(self):
        """Test that Random Forest can be trained on binary shot data"""
        # This test will fail until we implement the training script
        pytest.skip("Will implement after data loading")


class TestXGBoostBinary:
    """Test XGBoost binary classifier"""

    def test_xgboost_training(self):
        """Test that XGBoost can be trained on binary shot data"""
        pytest.skip("Will implement after Random Forest")


class TestMLPBinary:
    """Test MLP binary classifier"""

    def test_mlp_training(self):
        """Test that MLP can be trained on binary shot data"""
        pytest.skip("Will implement after XGBoost")


class TestBinaryEvaluation:
    """Test evaluation metrics for binary classification"""

    def test_roc_auc_calculation(self):
        """Test ROC-AUC metric calculation"""
        pytest.skip("Will implement after models are trained")

    def test_precision_recall_curve(self):
        """Test Precision-Recall curve generation"""
        pytest.skip("Will implement after models are trained")
