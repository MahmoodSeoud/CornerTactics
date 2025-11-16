#!/usr/bin/env python3
"""
Tests for model evaluation and analysis script.

Following TDD principles - these tests verify:
1. Model loading functionality
2. Confusion matrix generation
3. Feature importance extraction
4. Error analysis logic
5. Report generation
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Import functions from the script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module by loading it directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluate_models",
    os.path.join(os.path.dirname(__file__), '..', 'scripts', '06_evaluate_models.py')
)
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


# Fixtures

@pytest.fixture
def sample_test_data():
    """Create small synthetic test dataset."""
    np.random.seed(42)
    n_samples = 50

    X_test = np.random.randn(n_samples, 27)
    y_test = np.random.choice([0, 1, 2, 3], n_samples)

    return X_test, y_test


@pytest.fixture
def sample_models(sample_test_data):
    """Create and train simple models for testing."""
    X_test, y_test = sample_test_data

    # Create simple training data
    X_train = np.random.randn(100, 27)
    y_train = np.random.choice([0, 1, 2, 3], 100)

    # Train simple models
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)

    return {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }


@pytest.fixture
def sample_label_encoder():
    """Create a label encoder with 4 classes."""
    encoder = LabelEncoder()
    encoder.fit(['Ball Receipt', 'Clearance', 'Goalkeeper', 'Other'])
    return encoder


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Test Model Loading

def test_load_models_returns_all_models(temp_dir):
    """Test that load_models loads all expected models."""
    # Create dummy models
    rf_model = RandomForestClassifier(n_estimators=5)
    xgb_model = XGBClassifier(n_estimators=5)

    # Save them
    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
    with open(os.path.join(temp_dir, "models", "random_forest.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(temp_dir, "models", "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    # Load them
    models = em.load_models(os.path.join(temp_dir, "models"))

    assert 'Random Forest' in models, "Should load Random Forest model"
    assert 'XGBoost' in models, "Should load XGBoost model"
    assert isinstance(models['Random Forest'], RandomForestClassifier)
    assert isinstance(models['XGBoost'], XGBClassifier)


def test_load_label_encoder(temp_dir):
    """Test that label encoder can be loaded."""
    encoder = LabelEncoder()
    encoder.fit(['Ball Receipt', 'Clearance', 'Goalkeeper', 'Other'])

    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
    with open(os.path.join(temp_dir, "models", "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    loaded_encoder = em.load_label_encoder(os.path.join(temp_dir, "models"))

    assert len(loaded_encoder.classes_) == 4, "Should have 4 classes"
    assert 'Ball Receipt' in loaded_encoder.classes_


# Test Confusion Matrix Generation

def test_generate_confusion_matrices_creates_plots(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Test that confusion matrices are generated for all models."""
    X_test, y_test = sample_test_data

    output_dir = os.path.join(temp_dir, "results", "confusion_matrices")

    em.generate_confusion_matrices(sample_models, X_test, y_test, sample_label_encoder, output_dir)

    assert os.path.exists(os.path.join(output_dir, "random_forest_confusion_matrix.png"))
    assert os.path.exists(os.path.join(output_dir, "xgboost_confusion_matrix.png"))


def test_confusion_matrix_has_correct_shape(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Test that confusion matrix has correct dimensions."""
    X_test, y_test = sample_test_data

    output_dir = os.path.join(temp_dir, "results", "confusion_matrices")
    cm_data = em.generate_confusion_matrices(sample_models, X_test, y_test, sample_label_encoder, output_dir)

    # Check that confusion matrices have correct shape
    for model_name, cm in cm_data.items():
        assert cm.shape == (4, 4), f"{model_name} confusion matrix should be 4x4"


# Test Feature Importance Extraction

def test_extract_feature_importance_for_tree_models(sample_models):
    """Test that feature importance is extracted from tree-based models."""
    feature_names = [f'feature_{i}' for i in range(27)]

    importance_dict = em.extract_feature_importance(sample_models, feature_names)

    assert 'Random Forest' in importance_dict, "Should have RF importance"
    assert 'XGBoost' in importance_dict, "Should have XGBoost importance"
    assert len(importance_dict['Random Forest']) == 27, "Should have 27 features"


def test_feature_importance_plot_created(sample_models, temp_dir):
    """Test that feature importance plot is created."""
    feature_names = [f'feature_{i}' for i in range(27)]
    output_path = os.path.join(temp_dir, "feature_importance.png")

    importance_dict = em.extract_feature_importance(sample_models, feature_names)
    em.plot_feature_importance(importance_dict, output_path)

    assert os.path.exists(output_path), "Should create feature importance plot"


# Test Error Analysis

def test_find_most_confused_pairs(sample_models, sample_test_data, sample_label_encoder):
    """Test that most confused class pairs are identified."""
    X_test, y_test = sample_test_data

    confused_pairs = em.find_most_confused_pairs(sample_models, X_test, y_test, sample_label_encoder)

    assert isinstance(confused_pairs, dict), "Should return dict"
    for model_name, pairs in confused_pairs.items():
        assert isinstance(pairs, list), "Should have list of confused pairs"
        # Each pair should be a tuple (class1, class2, count)
        if len(pairs) > 0:
            assert len(pairs[0]) == 3, "Each pair should have (class1, class2, count)"


def test_analyze_misclassifications(sample_models, sample_test_data, sample_label_encoder):
    """Test that misclassification analysis is performed."""
    X_test, y_test = sample_test_data
    feature_names = [f'feature_{i}' for i in range(27)]

    analysis = em.analyze_misclassifications(sample_models, X_test, y_test, sample_label_encoder, feature_names)

    assert isinstance(analysis, dict), "Should return dict"
    for model_name in sample_models.keys():
        assert model_name in analysis, f"Should have analysis for {model_name}"


# Test Report Generation

def test_generate_report_creates_markdown_file(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Test that evaluation report markdown file is created."""
    X_test, y_test = sample_test_data
    feature_names = [f'feature_{i}' for i in range(27)]
    output_path = os.path.join(temp_dir, "evaluation_report.md")

    em.generate_evaluation_report(
        sample_models,
        X_test,
        y_test,
        sample_label_encoder,
        feature_names,
        output_path
    )

    assert os.path.exists(output_path), "Should create evaluation report"


def test_report_contains_key_sections(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Test that report contains all key sections."""
    X_test, y_test = sample_test_data
    feature_names = [f'feature_{i}' for i in range(27)]
    output_path = os.path.join(temp_dir, "evaluation_report.md")

    em.generate_evaluation_report(
        sample_models,
        X_test,
        y_test,
        sample_label_encoder,
        feature_names,
        output_path
    )

    with open(output_path, 'r') as f:
        content = f.read()

    assert "# Model Evaluation Report" in content
    assert "## Test Set Performance" in content
    assert "## Confusion Matrices" in content
    assert "## Feature Importance" in content
    assert "## Error Analysis" in content


# Test Per-Class F1 Comparison

def test_plot_per_class_f1_comparison(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Test that per-class F1 comparison plot is created."""
    X_test, y_test = sample_test_data
    output_path = os.path.join(temp_dir, "per_class_f1.png")

    em.plot_per_class_f1(sample_models, X_test, y_test, sample_label_encoder, output_path)

    assert os.path.exists(output_path), "Should create per-class F1 plot"


# Integration Test

def test_full_evaluation_pipeline(sample_models, sample_test_data, sample_label_encoder, temp_dir):
    """Integration test: full evaluation pipeline."""
    X_test, y_test = sample_test_data
    feature_names = [f'feature_{i}' for i in range(27)]

    # Create output directories
    results_dir = os.path.join(temp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate all outputs
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    em.generate_confusion_matrices(sample_models, X_test, y_test, sample_label_encoder, cm_dir)

    importance_dict = em.extract_feature_importance(sample_models, feature_names)
    em.plot_feature_importance(importance_dict, os.path.join(results_dir, "feature_importance.png"))

    em.plot_per_class_f1(sample_models, X_test, y_test, sample_label_encoder,
                         os.path.join(results_dir, "per_class_f1.png"))

    em.generate_evaluation_report(
        sample_models,
        X_test,
        y_test,
        sample_label_encoder,
        feature_names,
        os.path.join(results_dir, "evaluation_report.md")
    )

    # Verify all outputs exist
    assert os.path.exists(os.path.join(cm_dir, "random_forest_confusion_matrix.png"))
    assert os.path.exists(os.path.join(cm_dir, "xgboost_confusion_matrix.png"))
    assert os.path.exists(os.path.join(results_dir, "feature_importance.png"))
    assert os.path.exists(os.path.join(results_dir, "per_class_f1.png"))
    assert os.path.exists(os.path.join(results_dir, "evaluation_report.md"))
