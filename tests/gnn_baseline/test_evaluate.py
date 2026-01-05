"""Tests for evaluation module with statistical rigor.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import numpy as np


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_bootstrap_ci_returns_tuple(self):
        """Bootstrap CI should return (lower, upper) tuple."""
        from experiments.gnn_baseline.evaluate import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.9])

        lower, upper = bootstrap_ci(y_true, y_pred, metric='auc', n_bootstrap=100)

        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_lower_less_than_upper(self):
        """Lower bound should be less than or equal to upper bound."""
        from experiments.gnn_baseline.evaluate import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.9])

        lower, upper = bootstrap_ci(y_true, y_pred, metric='auc', n_bootstrap=100)

        assert lower <= upper

    def test_bootstrap_ci_bounds_valid(self):
        """CI bounds should be in valid range [0, 1] for AUC."""
        from experiments.gnn_baseline.evaluate import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.9])

        lower, upper = bootstrap_ci(y_true, y_pred, metric='auc', n_bootstrap=100)

        assert 0 <= lower <= 1
        assert 0 <= upper <= 1


class TestPermutationTest:
    """Test permutation test for significance."""

    def test_permutation_test_returns_pvalue(self):
        """Permutation test should return p-value."""
        from experiments.gnn_baseline.evaluate import permutation_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.9])

        p_value = permutation_test(y_true, y_pred, metric='auc', n_permutations=100)

        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_permutation_test_perfect_predictions(self):
        """Perfect predictions should have low p-value."""
        from experiments.gnn_baseline.evaluate import permutation_test

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])

        p_value = permutation_test(y_true, y_pred, metric='auc', n_permutations=100)

        # Should be significantly better than random
        assert p_value < 0.1

    def test_permutation_test_random_predictions(self):
        """Random predictions should have high p-value."""
        from experiments.gnn_baseline.evaluate import permutation_test

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Completely uninformative predictions
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        p_value = permutation_test(y_true, y_pred, metric='auc', n_permutations=100)

        # Should not be significantly better than random
        assert p_value > 0.05


class TestEvaluator:
    """Test Evaluator class for comprehensive evaluation."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6, 0.3, 0.9, 0.1, 0.2, 0.6, 0.8])
        return y_true, y_pred

    def test_evaluator_computes_auc(self, sample_predictions):
        """Evaluator should compute AUC score."""
        from experiments.gnn_baseline.evaluate import Evaluator

        y_true, y_pred = sample_predictions
        evaluator = Evaluator(y_true, y_pred)

        result = evaluator.evaluate()

        assert 'auc' in result
        assert 0 <= result['auc'] <= 1

    def test_evaluator_computes_ci(self, sample_predictions):
        """Evaluator should compute confidence intervals."""
        from experiments.gnn_baseline.evaluate import Evaluator

        y_true, y_pred = sample_predictions
        evaluator = Evaluator(y_true, y_pred)

        result = evaluator.evaluate(n_bootstrap=100)

        assert 'auc_ci_lower' in result
        assert 'auc_ci_upper' in result

    def test_evaluator_computes_pvalue(self, sample_predictions):
        """Evaluator should compute p-value vs random baseline."""
        from experiments.gnn_baseline.evaluate import Evaluator

        y_true, y_pred = sample_predictions
        evaluator = Evaluator(y_true, y_pred)

        result = evaluator.evaluate(n_permutations=100)

        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1

    def test_evaluator_significant_flag(self, sample_predictions):
        """Evaluator should indicate if result is statistically significant."""
        from experiments.gnn_baseline.evaluate import Evaluator

        y_true, y_pred = sample_predictions
        evaluator = Evaluator(y_true, y_pred)

        result = evaluator.evaluate(n_permutations=100, alpha=0.05)

        assert 'is_significant' in result
        assert isinstance(result['is_significant'], bool)


class TestModelEvaluation:
    """Test model evaluation on test set."""

    @pytest.fixture
    def sample_corners(self):
        """Create sample corner data."""
        corners = []
        for i in range(20):
            corner = {
                "match_id": str(i // 5),
                "event": {"id": f"corner-{i}", "location": [120.0, 0.0]},
                "freeze_frame": [
                    {"location": [100.0 + j, 30.0 + j * 2], "teammate": j < 5, "keeper": j == 9, "actor": j == 0}
                    for j in range(10)
                ],
                "shot_outcome": i % 2,
            }
            corners.append(corner)
        return corners

    @pytest.fixture
    def split_indices(self):
        """Create train/val/test split."""
        return {
            "train": list(range(0, 12)),
            "val": list(range(12, 16)),
            "test": list(range(16, 20)),
        }

    def test_evaluate_model(self, sample_corners, split_indices):
        """Should evaluate trained model on test set."""
        from experiments.gnn_baseline.train import Trainer, set_seed
        from experiments.gnn_baseline.evaluate import evaluate_model

        set_seed(42)
        trainer = Trainer(
            corners=sample_corners,
            train_indices=split_indices["train"],
            val_indices=split_indices["val"],
            test_indices=split_indices["test"],
            model_name='graphsage',
            hidden_channels=32,
            batch_size=4,
        )
        trainer.fit(epochs=2, verbose=False)

        results = evaluate_model(trainer)

        assert 'test_auc' in results
        assert 'test_auc_ci_lower' in results
        assert 'test_auc_ci_upper' in results
        assert 'test_p_value' in results


class TestResultsFormatting:
    """Test results formatting for thesis."""

    def test_format_results(self):
        """Should format results as thesis-ready string."""
        from experiments.gnn_baseline.evaluate import format_results

        results = {
            'auc': 0.65,
            'auc_ci_lower': 0.58,
            'auc_ci_upper': 0.72,
            'p_value': 0.003,
            'is_significant': True,
        }

        formatted = format_results(results)

        # Should contain key information
        assert '0.65' in formatted
        assert '0.58' in formatted or '0.72' in formatted
        assert 'p' in formatted.lower()
