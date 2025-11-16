#!/usr/bin/env python3
"""
Tests for baseline models training script.

Following TDD principles - these tests verify:
1. Data loading and preparation
2. Model training functions
3. Evaluation and metrics
4. Model persistence
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

# Import functions from the script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module by loading it directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_baseline_models",
    os.path.join(os.path.dirname(__file__), '..', 'scripts', '05_train_baseline_models.py')
)
tbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tbm)


# Fixtures

@pytest.fixture
def sample_features_df():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'match_id': [f'match_{i}' for i in range(n_samples)],
        'event_id': [f'event_{i}' for i in range(n_samples)],
        'outcome': np.random.choice(['Ball Receipt', 'Clearance', 'Goalkeeper', 'Other'], n_samples),
        'corner_side': np.random.randint(0, 2, n_samples),
        'period': np.random.randint(1, 3, n_samples),
        'minute': np.random.randint(0, 90, n_samples),
        'corner_x': np.random.uniform(115, 120, n_samples),
        'corner_y': np.random.uniform(0, 80, n_samples),
        'total_attacking': np.random.randint(5, 12, n_samples),
        'total_defending': np.random.randint(8, 15, n_samples),
        'attacking_in_box': np.random.randint(2, 8, n_samples),
        'defending_in_box': np.random.randint(6, 12, n_samples),
        'attacking_near_goal': np.random.randint(1, 5, n_samples),
        'defending_near_goal': np.random.randint(3, 8, n_samples),
        'attacking_density': np.random.uniform(0.005, 0.015, n_samples),
        'defending_density': np.random.uniform(0.008, 0.020, n_samples),
        'numerical_advantage': np.random.randint(-5, 2, n_samples),
        'attacker_defender_ratio': np.random.uniform(0.3, 0.9, n_samples),
        'attacking_centroid_x': np.random.uniform(108, 115, n_samples),
        'attacking_centroid_y': np.random.uniform(30, 50, n_samples),
        'defending_centroid_x': np.random.uniform(110, 116, n_samples),
        'defending_centroid_y': np.random.uniform(30, 50, n_samples),
        'defending_compactness': np.random.uniform(5, 15, n_samples),
        'defending_depth': np.random.uniform(8, 18, n_samples),
        'attacking_to_goal_dist': np.random.uniform(5, 15, n_samples),
        'defending_to_goal_dist': np.random.uniform(3, 10, n_samples),
        'pass_end_x': np.random.uniform(100, 118, n_samples),
        'pass_end_y': np.random.uniform(20, 60, n_samples),
        'pass_length': np.random.uniform(10, 50, n_samples),
        'pass_height': np.random.randint(0, 3, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_splits():
    """Create sample train/val/test split indices."""
    train_indices = np.arange(0, 60)
    val_indices = np.arange(60, 80)
    test_indices = np.arange(80, 100)
    return train_indices, val_indices, test_indices


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Test Data Loading

def test_prepare_features_returns_correct_shapes(sample_features_df):
    """Test that prepare_features returns arrays of correct shape."""
    X, y, label_encoder, feature_cols = tbm.prepare_features(sample_features_df)

    assert X.shape[0] == len(sample_features_df), "X should have same number of rows as input"
    assert X.shape[1] == 27, "X should have 27 feature columns"
    assert len(y) == len(sample_features_df), "y should have same length as input"
    assert len(label_encoder.classes_) == 4, "Should have 4 outcome classes"


def test_prepare_features_excludes_metadata(sample_features_df):
    """Test that metadata columns are excluded from features."""
    X, y, label_encoder, feature_cols = tbm.prepare_features(sample_features_df)

    assert 'match_id' not in feature_cols, "match_id should be excluded"
    assert 'event_id' not in feature_cols, "event_id should be excluded"
    assert 'outcome' not in feature_cols, "outcome should be excluded"
    assert 'corner_side' in feature_cols, "corner_side should be included"


def test_prepare_features_encodes_labels(sample_features_df):
    """Test that outcome labels are properly encoded."""
    X, y, label_encoder, feature_cols = tbm.prepare_features(sample_features_df)

    # Check that y is numeric
    assert np.issubdtype(y.dtype, np.integer), "y should be integer encoded"

    # Check that all values are in valid range
    assert y.min() >= 0, "Encoded labels should be >= 0"
    assert y.max() < len(label_encoder.classes_), "Encoded labels should be < num_classes"

    # Check that we can decode back
    decoded = label_encoder.inverse_transform(y)
    assert set(decoded) == set(sample_features_df['outcome']), "Should be able to decode back to original labels"


# Test Class Weight Calculation

def test_calculate_class_weights_returns_dict():
    """Test that calculate_class_weights returns a dictionary."""
    y_train = np.array([0, 0, 0, 0, 1, 1, 2])
    weights = tbm.calculate_class_weights(y_train)

    assert isinstance(weights, dict), "Should return a dictionary"
    assert len(weights) == 3, "Should have weights for 3 classes"


def test_calculate_class_weights_balances_classes():
    """Test that class weights are inversely proportional to frequency."""
    y_train = np.array([0, 0, 0, 0, 1, 1, 2])  # 4:2:1 ratio
    weights = tbm.calculate_class_weights(y_train)

    # Class 0 appears 4 times, should have lowest weight
    # Class 2 appears 1 time, should have highest weight
    assert weights[2] > weights[1] > weights[0], "Weights should be inversely proportional to frequency"


# Test Model Training Functions

def test_train_random_forest_returns_model_and_metrics(sample_features_df, sample_splits):
    """Test that train_random_forest returns a model and metrics dict."""
    X, y, _, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, _ = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model, metrics = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)

    # Check model type
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "Should return a RandomForestClassifier"

    # Check metrics
    assert isinstance(metrics, dict), "Should return metrics as dict"
    assert 'train_f1_macro' in metrics, "Should have train_f1_macro"
    assert 'val_f1_macro' in metrics, "Should have val_f1_macro"
    assert 0 <= metrics['train_f1_macro'] <= 1, "F1 score should be between 0 and 1"


def test_train_xgboost_returns_model_and_metrics(sample_features_df, sample_splits):
    """Test that train_xgboost returns a model and metrics dict."""
    X, y, _, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, _ = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model, metrics = tbm.train_xgboost(X_train, y_train, X_val, y_val, n_classes=4)

    # Check model type
    from xgboost import XGBClassifier
    assert isinstance(model, XGBClassifier), "Should return an XGBClassifier"

    # Check metrics
    assert isinstance(metrics, dict), "Should return metrics as dict"
    assert 'train_f1_macro' in metrics, "Should have train_f1_macro"
    assert 'val_f1_macro' in metrics, "Should have val_f1_macro"


def test_train_mlp_returns_model_scaler_and_metrics(sample_features_df, sample_splits):
    """Test that train_mlp returns a model, scaler, and metrics dict."""
    X, y, _, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, _ = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model, scaler, metrics = tbm.train_mlp(X_train, y_train, X_val, y_val, n_classes=4)

    # Check model type
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    assert isinstance(model, MLPClassifier), "Should return an MLPClassifier"
    assert isinstance(scaler, StandardScaler), "Should return a StandardScaler"

    # Check metrics
    assert isinstance(metrics, dict), "Should return metrics as dict"
    assert 'train_f1_macro' in metrics, "Should have train_f1_macro"
    assert 'val_f1_macro' in metrics, "Should have val_f1_macro"


# Test Model Evaluation

def test_evaluate_on_test_returns_results_for_all_models(sample_features_df, sample_splits):
    """Test that evaluate_on_test returns results for all models."""
    X, y, label_encoder, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, test_idx = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train models
    rf_model, _ = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)
    xgb_model, _ = tbm.train_xgboost(X_train, y_train, X_val, y_val, n_classes=4)
    mlp_model, scaler, _ = tbm.train_mlp(X_train, y_train, X_val, y_val, n_classes=4)

    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'MLP': mlp_model
    }

    results = tbm.evaluate_on_test(models, X_test, y_test, label_encoder, scaler)

    assert len(results) == 3, "Should have results for 3 models"
    assert 'Random Forest' in results, "Should have Random Forest results"
    assert 'XGBoost' in results, "Should have XGBoost results"
    assert 'MLP' in results, "Should have MLP results"

    # Check structure of results
    for model_name, result in results.items():
        assert 'test_f1_macro' in result, f"{model_name} should have test_f1_macro"
        assert 'test_accuracy' in result, f"{model_name} should have test_accuracy"
        assert 'confusion_matrix' in result, f"{model_name} should have confusion_matrix"


def test_evaluate_on_test_metrics_are_valid(sample_features_df, sample_splits):
    """Test that evaluation metrics are in valid ranges."""
    X, y, label_encoder, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, test_idx = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train a simple model
    rf_model, _ = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)

    models = {'Random Forest': rf_model}
    results = tbm.evaluate_on_test(models, X_test, y_test, label_encoder, None)

    rf_results = results['Random Forest']
    assert 0 <= rf_results['test_f1_macro'] <= 1, "F1 macro should be between 0 and 1"
    assert 0 <= rf_results['test_accuracy'] <= 1, "Accuracy should be between 0 and 1"


# Test Model Persistence

def test_save_models_creates_files(sample_features_df, sample_splits, temp_dir):
    """Test that save_models creates all expected files."""
    X, y, label_encoder, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, _ = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train models
    rf_model, rf_metrics = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)
    xgb_model, xgb_metrics = tbm.train_xgboost(X_train, y_train, X_val, y_val, n_classes=4)
    mlp_model, scaler, mlp_metrics = tbm.train_mlp(X_train, y_train, X_val, y_val, n_classes=4)

    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'MLP': mlp_model
    }

    results = {
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
        'MLP': mlp_metrics
    }

    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        tbm.save_models(models, label_encoder, scaler, results)

        # Check that files exist
        assert os.path.exists("models/random_forest.pkl"), "Should create random_forest.pkl"
        assert os.path.exists("models/xgboost.pkl"), "Should create xgboost.pkl"
        assert os.path.exists("models/mlp.pkl"), "Should create mlp.pkl"
        assert os.path.exists("models/label_encoder.pkl"), "Should create label_encoder.pkl"
        assert os.path.exists("models/feature_scaler.pkl"), "Should create feature_scaler.pkl"
        assert os.path.exists("results/baseline_metrics.json"), "Should create baseline_metrics.json"
    finally:
        os.chdir(original_dir)


def test_saved_models_can_be_loaded(sample_features_df, sample_splits, temp_dir):
    """Test that saved models can be loaded and used for prediction."""
    X, y, label_encoder, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, test_idx = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train a model
    rf_model, rf_metrics = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)

    models = {'Random Forest': rf_model}
    results = {'Random Forest': rf_metrics}

    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        tbm.save_models(models, label_encoder, None, results)

        # Load the model
        with open("models/random_forest.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # Load the label encoder
        with open("models/label_encoder.pkl", "rb") as f:
            loaded_encoder = pickle.load(f)

        # Make predictions
        predictions = loaded_model.predict(X_test)

        # Check that predictions are valid
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        assert all(0 <= p < 4 for p in predictions), "Predictions should be valid class indices"
    finally:
        os.chdir(original_dir)


def test_saved_metrics_have_correct_structure(sample_features_df, sample_splits, temp_dir):
    """Test that saved metrics JSON has the correct structure."""
    X, y, label_encoder, _ = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, _ = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train models
    rf_model, rf_metrics = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes=4)

    models = {'Random Forest': rf_model}
    results = {'Random Forest': rf_metrics}

    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        tbm.save_models(models, label_encoder, None, results)

        # Load the JSON
        with open("results/baseline_metrics.json", "r") as f:
            metrics = json.load(f)

        # Check structure
        assert 'Random Forest' in metrics, "Should have Random Forest key"
        assert 'train_f1_macro' in metrics['Random Forest'], "Should have train_f1_macro"
        assert 'val_f1_macro' in metrics['Random Forest'], "Should have val_f1_macro"
    finally:
        os.chdir(original_dir)


# Integration Test

def test_full_training_pipeline(sample_features_df, sample_splits, temp_dir):
    """Integration test: full training pipeline with synthetic data."""
    X, y, label_encoder, feature_cols = tbm.prepare_features(sample_features_df)
    train_idx, val_idx, test_idx = sample_splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    n_classes = len(label_encoder.classes_)

    # Train all models
    rf_model, rf_metrics = tbm.train_random_forest(X_train, y_train, X_val, y_val, n_classes)
    xgb_model, xgb_metrics = tbm.train_xgboost(X_train, y_train, X_val, y_val, n_classes)
    mlp_model, scaler, mlp_metrics = tbm.train_mlp(X_train, y_train, X_val, y_val, n_classes)

    # Collect models
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'MLP': mlp_model
    }

    # Evaluate on test set
    test_results = tbm.evaluate_on_test(models, X_test, y_test, label_encoder, scaler)

    # Merge results
    all_results = {}
    for model_name in models.keys():
        if model_name == 'Random Forest':
            all_results[model_name] = {**rf_metrics, **test_results[model_name]}
        elif model_name == 'XGBoost':
            all_results[model_name] = {**xgb_metrics, **test_results[model_name]}
        elif model_name == 'MLP':
            all_results[model_name] = {**mlp_metrics, **test_results[model_name]}

    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Save everything
        tbm.save_models(models, label_encoder, scaler, all_results)

        # Verify all outputs exist
        assert os.path.exists("models/random_forest.pkl")
        assert os.path.exists("models/xgboost.pkl")
        assert os.path.exists("models/mlp.pkl")
        assert os.path.exists("models/label_encoder.pkl")
        assert os.path.exists("models/feature_scaler.pkl")
        assert os.path.exists("results/baseline_metrics.json")

        # Verify metrics are reasonable
        for model_name, metrics in all_results.items():
            assert 0 <= metrics['val_f1_macro'] <= 1
            assert 0 <= metrics['test_f1_macro'] <= 1
            assert 0 <= metrics['test_accuracy'] <= 1
    finally:
        os.chdir(original_dir)
