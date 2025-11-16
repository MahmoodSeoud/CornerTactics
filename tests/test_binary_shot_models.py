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

    def test_random_forest_model_exists(self):
        """Test that Random Forest model was saved"""
        model_path = Path("models/binary/random_forest.pkl")
        assert model_path.exists(), "Random Forest model should exist"

    def test_random_forest_can_load_and_predict(self):
        """Test that we can load and use the Random Forest model"""
        import pickle

        # Load model
        with open("models/binary/random_forest.pkl", 'rb') as f:
            rf = pickle.load(f)

        # Load test data
        with open("data/processed/corners_with_shot_labels.json") as f:
            shot_data = json.load(f)
        shot_labels = [{'event_id': c['event']['id'], 'shot_outcome': c['shot_outcome']}
                      for c in shot_data]
        shot_df = pd.DataFrame(shot_labels)

        features_df = pd.read_csv("data/processed/corners_with_features.csv")
        merged = pd.merge(features_df, shot_df, on='event_id', how='inner')

        test_indices = pd.read_csv("data/processed/test_indices.csv")['index'].values
        exclude_cols = ['match_id', 'event_id', 'outcome', 'shot_outcome']
        feature_cols = [col for col in merged.columns if col not in exclude_cols]

        X_test = merged.iloc[test_indices][feature_cols].values
        y_test = merged.iloc[test_indices]['shot_outcome'].values

        # Make predictions
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)

        assert len(y_pred) == len(y_test), "Predictions should match test set size"
        assert y_proba.shape == (len(y_test), 2), "Probabilities should be binary"


class TestXGBoostBinary:
    """Test XGBoost binary classifier"""

    def test_xgboost_model_exists(self):
        """Test that XGBoost model was saved"""
        model_path = Path("models/binary/xgboost.json")
        assert model_path.exists(), "XGBoost model should exist"

    def test_xgboost_can_load_and_predict(self):
        """Test that we can load and use the XGBoost model"""
        import xgboost as xgb

        # Load model
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/binary/xgboost.json")

        # Load test data
        with open("data/processed/corners_with_shot_labels.json") as f:
            shot_data = json.load(f)
        shot_labels = [{'event_id': c['event']['id'], 'shot_outcome': c['shot_outcome']}
                      for c in shot_data]
        shot_df = pd.DataFrame(shot_labels)

        features_df = pd.read_csv("data/processed/corners_with_features.csv")
        merged = pd.merge(features_df, shot_df, on='event_id', how='inner')

        test_indices = pd.read_csv("data/processed/test_indices.csv")['index'].values
        exclude_cols = ['match_id', 'event_id', 'outcome', 'shot_outcome']
        feature_cols = [col for col in merged.columns if col not in exclude_cols]

        X_test = merged.iloc[test_indices][feature_cols].values

        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)

        assert len(y_pred) > 0, "Should make predictions"
        assert y_proba.shape[1] == 2, "Probabilities should be binary"


class TestMLPBinary:
    """Test MLP binary classifier"""

    def test_mlp_model_exists(self):
        """Test that MLP model was saved"""
        model_path = Path("models/binary/mlp.pkl")
        scaler_path = Path("models/binary/feature_scaler.pkl")
        assert model_path.exists(), "MLP model should exist"
        assert scaler_path.exists(), "Feature scaler should exist"

    def test_mlp_can_load_and_predict(self):
        """Test that we can load and use the MLP model"""
        import pickle

        # Load model and scaler
        with open("models/binary/mlp.pkl", 'rb') as f:
            mlp = pickle.load(f)
        with open("models/binary/feature_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        # Load test data
        with open("data/processed/corners_with_shot_labels.json") as f:
            shot_data = json.load(f)
        shot_labels = [{'event_id': c['event']['id'], 'shot_outcome': c['shot_outcome']}
                      for c in shot_data]
        shot_df = pd.DataFrame(shot_labels)

        features_df = pd.read_csv("data/processed/corners_with_features.csv")
        merged = pd.merge(features_df, shot_df, on='event_id', how='inner')

        test_indices = pd.read_csv("data/processed/test_indices.csv")['index'].values
        exclude_cols = ['match_id', 'event_id', 'outcome', 'shot_outcome']
        feature_cols = [col for col in merged.columns if col not in exclude_cols]

        X_test = merged.iloc[test_indices][feature_cols].values

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        y_pred = mlp.predict(X_test_scaled)
        y_proba = mlp.predict_proba(X_test_scaled)

        assert len(y_pred) > 0, "Should make predictions"
        assert y_proba.shape[1] == 2, "Probabilities should be binary"


class TestBinaryEvaluation:
    """Test evaluation metrics for binary classification"""

    def test_metrics_file_exists(self):
        """Test that metrics file was created"""
        metrics_path = Path("results/binary_metrics.json")
        assert metrics_path.exists(), "Metrics file should exist"

    def test_metrics_content(self):
        """Test that metrics contain expected fields"""
        with open("results/binary_metrics.json") as f:
            metrics = json.load(f)

        assert 'random_forest' in metrics, "Should have Random Forest metrics"
        assert 'xgboost' in metrics, "Should have XGBoost metrics"
        assert 'mlp' in metrics, "Should have MLP metrics"

        for model in ['random_forest', 'xgboost', 'mlp']:
            assert 'accuracy' in metrics[model], f"{model} should have accuracy"
            assert 'precision' in metrics[model], f"{model} should have precision"
            assert 'recall' in metrics[model], f"{model} should have recall"
            assert 'f1' in metrics[model], f"{model} should have f1"
            assert 'roc_auc' in metrics[model], f"{model} should have roc_auc"
            assert 'pr_auc' in metrics[model], f"{model} should have pr_auc"

    def test_confusion_matrices_exist(self):
        """Test that confusion matrix plots were created"""
        cm_dir = Path("results/confusion_matrices_binary")
        assert cm_dir.exists(), "Confusion matrices directory should exist"

        assert (cm_dir / "random_forest.png").exists(), "RF confusion matrix should exist"
        assert (cm_dir / "xgboost.png").exists(), "XGBoost confusion matrix should exist"
        assert (cm_dir / "mlp.png").exists(), "MLP confusion matrix should exist"
