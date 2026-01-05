"""Tests for transfer learning analysis.

TDD: Test whether offside-predictive features improve shot prediction.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_corners():
    """Sample corners with varied positions."""
    corners = []
    np.random.seed(42)
    for i in range(50):
        # Generate semi-realistic freeze frames
        freeze_frame = []
        # Attackers
        for j in range(5):
            freeze_frame.append({
                "location": [100 + np.random.uniform(0, 15), 20 + np.random.uniform(0, 40)],
                "teammate": True,
                "keeper": False,
                "actor": j == 0,
            })
        # Defenders
        for j in range(4):
            freeze_frame.append({
                "location": [105 + np.random.uniform(0, 10), 25 + np.random.uniform(0, 30)],
                "teammate": False,
                "keeper": False,
                "actor": False,
            })
        # Goalkeeper
        freeze_frame.append({
            "location": [120.0, 40.0],
            "teammate": False,
            "keeper": True,
            "actor": False,
        })

        corners.append({
            "match_id": str(i),
            "event": {"id": f"c{i}", "location": [120.0, 0.0 if i % 2 == 0 else 80.0]},
            "freeze_frame": freeze_frame,
            "shot_outcome": i % 3 == 0,  # ~33% shots
        })

    return corners


class TestFeatureAugmentation:
    """Test adding offside features to existing datasets."""

    def test_augment_with_offside_features(self, sample_corners):
        """Should add offside features to corner data."""
        from experiments.offside_analysis.transfer_learning import augment_with_offside_features

        augmented = augment_with_offside_features(sample_corners)

        assert len(augmented) == len(sample_corners)
        assert 'offside_features' in augmented[0]
        assert 'attackers_beyond_defender' in augmented[0]['offside_features']

    def test_create_feature_matrix(self, sample_corners):
        """Should create feature matrix from corners."""
        from experiments.offside_analysis.transfer_learning import create_feature_matrix

        X, y = create_feature_matrix(sample_corners)

        assert X.shape[0] == len(sample_corners)
        assert len(y) == len(sample_corners)
        assert X.shape[1] > 0  # Has features


class TestClassifierExperiments:
    """Test classifier training with offside features."""

    def test_train_baseline_classifier(self, sample_corners):
        """Should train baseline classifier without offside features."""
        from experiments.offside_analysis.transfer_learning import train_baseline_classifier

        model, metrics = train_baseline_classifier(sample_corners)

        assert model is not None
        assert 'auc' in metrics
        assert 0 <= metrics['auc'] <= 1

    def test_train_with_offside_features(self, sample_corners):
        """Should train classifier with offside features."""
        from experiments.offside_analysis.transfer_learning import train_with_offside_features

        model, metrics = train_with_offside_features(sample_corners)

        assert model is not None
        assert 'auc' in metrics

    def test_compare_classifiers(self, sample_corners):
        """Should compare baseline vs offside-augmented classifier."""
        from experiments.offside_analysis.transfer_learning import compare_classifiers

        comparison = compare_classifiers(sample_corners)

        assert 'baseline_auc' in comparison
        assert 'augmented_auc' in comparison
        assert 'improvement' in comparison


class TestHierarchicalClassifier:
    """Test hierarchical classification approach."""

    def test_train_hierarchical_classifier(self, sample_corners):
        """Should train two-stage hierarchical classifier."""
        from experiments.offside_analysis.transfer_learning import train_hierarchical_classifier

        model, metrics = train_hierarchical_classifier(sample_corners)

        assert model is not None
        assert 'stage1_auc' in metrics or 'overall_auc' in metrics


class TestFeatureImportance:
    """Test feature importance analysis."""

    def test_compute_feature_importance(self, sample_corners):
        """Should compute importance of each feature."""
        from experiments.offside_analysis.transfer_learning import compute_feature_importance

        importance = compute_feature_importance(sample_corners)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Values should be numeric
        for key, val in importance.items():
            assert isinstance(val, (int, float))

    def test_rank_offside_features(self, sample_corners):
        """Should rank offside features by importance."""
        from experiments.offside_analysis.transfer_learning import rank_offside_features

        ranking = rank_offside_features(sample_corners)

        assert isinstance(ranking, list)
        assert len(ranking) > 0
        # Should be sorted by importance (descending)
        if len(ranking) > 1:
            assert ranking[0][1] >= ranking[1][1]


class TestStatisticalAnalysis:
    """Test statistical analysis of feature differences."""

    def test_compute_feature_statistics(self, sample_corners):
        """Should compute statistics of features by outcome."""
        from experiments.offside_analysis.transfer_learning import compute_feature_statistics

        stats = compute_feature_statistics(sample_corners)

        assert 'shot' in stats
        assert 'no_shot' in stats
        assert 'mean' in stats['shot']
        assert 'std' in stats['shot']

    def test_compute_feature_significance(self, sample_corners):
        """Should test statistical significance of feature differences."""
        from experiments.offside_analysis.transfer_learning import compute_feature_significance

        significance = compute_feature_significance(sample_corners)

        assert isinstance(significance, dict)
        # Each feature should have p-value and effect size
        for feature, result in significance.items():
            assert 'p_value' in result
            assert 't_statistic' in result or 'statistic' in result
