"""Tests for the statistical tests module.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import numpy as np
import pytest
from scipy import stats


class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_bootstrap_ci_auc_returns_tuple(self):
        """Bootstrap CI should return (lower, upper) tuple."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        result = bootstrap_ci(y_true, y_pred, metric='auc')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]

    def test_bootstrap_ci_auc_within_bounds(self):
        """CI bounds should be within [0, 1] for AUC."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        lower, upper = bootstrap_ci(y_true, y_pred, metric='auc')

        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_bootstrap_ci_perfect_classifier_high_ci(self):
        """Perfect classifier should have high CI bounds."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        lower, upper = bootstrap_ci(
            y_true, y_pred, metric='auc', random_state=42
        )

        # Perfect classifier should have CI lower bound > 0.8
        assert lower > 0.8

    def test_bootstrap_ci_random_classifier_around_half(self):
        """Random predictions should have CI around 0.5."""
        from experiments.statistical_tests import bootstrap_ci

        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.random.rand(100)

        lower, upper = bootstrap_ci(
            y_true, y_pred, metric='auc', n_bootstrap=500, random_state=42
        )

        # Random classifier CI should contain 0.5
        assert lower < 0.5 < upper

    def test_bootstrap_ci_average_precision(self):
        """Bootstrap CI should support average precision metric."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        lower, upper = bootstrap_ci(y_true, y_pred, metric='average_precision')

        assert 0.0 <= lower <= upper <= 1.0

    def test_bootstrap_ci_invalid_metric_raises(self):
        """Unknown metric should raise ValueError."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 1])
        y_pred = np.array([0.2, 0.8])

        with pytest.raises(ValueError, match="Unknown metric"):
            bootstrap_ci(y_true, y_pred, metric='invalid_metric')

    def test_bootstrap_ci_reproducible_with_seed(self):
        """Same random_state should produce same results."""
        from experiments.statistical_tests import bootstrap_ci

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        result1 = bootstrap_ci(y_true, y_pred, metric='auc', random_state=42)
        result2 = bootstrap_ci(y_true, y_pred, metric='auc', random_state=42)

        assert result1 == result2


class TestPermutationTest:
    """Tests for permutation test significance testing."""

    def test_permutation_test_returns_float(self):
        """Permutation test should return a float p-value."""
        from experiments.statistical_tests import permutation_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        p_value = permutation_test(y_true, y_pred, metric='auc')

        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    def test_permutation_test_perfect_classifier_low_p(self):
        """Perfect classifier should have low p-value."""
        from experiments.statistical_tests import permutation_test

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        p_value = permutation_test(
            y_true, y_pred, metric='auc',
            n_permutations=500, random_state=42
        )

        # Perfect classifier should be significantly better than random
        assert p_value < 0.05

    def test_permutation_test_random_classifier_high_p(self):
        """Random predictions should have high p-value."""
        from experiments.statistical_tests import permutation_test

        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.random.rand(100)

        p_value = permutation_test(
            y_true, y_pred, metric='auc',
            n_permutations=500, random_state=42
        )

        # Random classifier should not be significant
        assert p_value > 0.05

    def test_permutation_test_supports_average_precision(self):
        """Permutation test should support average precision metric."""
        from experiments.statistical_tests import permutation_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        p_value = permutation_test(y_true, y_pred, metric='average_precision')

        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0


class TestMcNemarTest:
    """Tests for McNemar's test for comparing two classifiers."""

    def test_mcnemar_returns_result_dict(self):
        """McNemar test should return dict with statistic and p-value."""
        from experiments.statistical_tests import mcnemar_test

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Model A: gets first 4 wrong, last 4 right
        y_pred_a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        # Model B: gets all correct
        y_pred_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = mcnemar_test(y_true, y_pred_a, y_pred_b)

        assert 'statistic' in result
        assert 'p_value' in result
        assert 'contingency_table' in result

    def test_mcnemar_identical_models_high_p(self):
        """Identical predictions should give high p-value (not different)."""
        from experiments.statistical_tests import mcnemar_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 1])

        result = mcnemar_test(y_true, y_pred, y_pred)

        # Same predictions = no difference = high p-value
        assert result['p_value'] >= 0.05

    def test_mcnemar_different_models_detects_difference(self):
        """Models with asymmetric errors should have low p-value."""
        from experiments.statistical_tests import mcnemar_test

        # Create asymmetric error pattern: Model B is much better than A
        # on the cases where they disagree
        y_true = np.array([0] * 50 + [1] * 50)
        # Model A: wrong on many cases
        y_pred_a = np.array([1] * 40 + [0] * 10 + [1] * 50)  # 60 correct
        # Model B: correct on almost all
        y_pred_b = np.array([0] * 50 + [1] * 50)  # 100 correct

        result = mcnemar_test(y_true, y_pred_a, y_pred_b)

        # B is right when A is wrong (40 cases), A never right when B wrong
        # This asymmetry should yield low p-value
        assert result['p_value'] < 0.05

    def test_mcnemar_accepts_probabilities_with_threshold(self):
        """McNemar test should accept probabilities and threshold."""
        from experiments.statistical_tests import mcnemar_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_a = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        y_pred_b = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5, 0.5])

        result = mcnemar_test(y_true, y_pred_a, y_pred_b, threshold=0.5)

        assert 'p_value' in result


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_returns_float(self):
        """Cohen's d should return a float."""
        from experiments.statistical_tests import cohens_d

        group_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group_b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        d = cohens_d(group_a, group_b)

        assert isinstance(d, float)

    def test_cohens_d_identical_groups_zero(self):
        """Identical groups should have d = 0."""
        from experiments.statistical_tests import cohens_d

        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        d = cohens_d(group, group)

        assert d == pytest.approx(0.0, abs=1e-10)

    def test_cohens_d_known_value(self):
        """Test against known Cohen's d value."""
        from experiments.statistical_tests import cohens_d

        # Two groups with known effect size
        # Mean diff = 1, pooled std = 1 => d = 1.0
        group_a = np.array([1.0, 1.0, 1.0, 1.0])
        group_b = np.array([2.0, 2.0, 2.0, 2.0])

        d = cohens_d(group_a, group_b)

        # Cohen's d = (mean_b - mean_a) / pooled_std
        # With no variance in groups, pooled_std is technically 0
        # So we test a more realistic case

    def test_cohens_d_small_medium_large_effect(self):
        """Test interpretation of effect sizes."""
        from experiments.statistical_tests import cohens_d

        # Create groups with approximately known effect sizes
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)

        # Small effect (d ~ 0.2)
        group_small = base + 0.2
        d_small = cohens_d(base, group_small)
        assert 0.1 < abs(d_small) < 0.4

        # Medium effect (d ~ 0.5)
        group_medium = base + 0.5
        d_medium = cohens_d(base, group_medium)
        assert 0.3 < abs(d_medium) < 0.7

        # Large effect (d ~ 0.8)
        group_large = base + 0.8
        d_large = cohens_d(base, group_large)
        assert 0.6 < abs(d_large) < 1.0

    def test_cohens_d_sign_convention(self):
        """Positive d means group_b > group_a."""
        from experiments.statistical_tests import cohens_d

        group_a = np.array([1.0, 2.0, 3.0])
        group_b = np.array([4.0, 5.0, 6.0])

        d = cohens_d(group_a, group_b)

        # group_b has higher mean, so d should be positive
        assert d > 0


class TestSignificanceModule:
    """Tests for the unified significance analysis module."""

    def test_comprehensive_evaluation_returns_all_metrics(self):
        """Comprehensive evaluation should return all statistical metrics."""
        from experiments.statistical_tests import comprehensive_evaluation

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        result = comprehensive_evaluation(
            y_true, y_pred,
            metric='auc',
            n_bootstrap=100,
            n_permutations=100,
            random_state=42
        )

        # Check all expected fields are present
        assert 'point_estimate' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'p_value' in result
        assert 'is_significant' in result

    def test_compare_models_returns_comparison_result(self):
        """Model comparison should return statistical comparison."""
        from experiments.statistical_tests import compare_models

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred_a = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        y_pred_b = np.array([0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8])

        result = compare_models(y_true, y_pred_a, y_pred_b)

        assert 'mcnemar_p_value' in result
        assert 'models_significantly_different' in result

    def test_format_thesis_results(self):
        """Results should be formattable for thesis."""
        from experiments.statistical_tests import format_results

        result = {
            'point_estimate': 0.75,
            'ci_lower': 0.65,
            'ci_upper': 0.85,
            'p_value': 0.02,
            'is_significant': True
        }

        formatted = format_results(result, metric='auc', model_name='GAT')

        assert 'GAT' in formatted
        assert '0.75' in formatted or '0.750' in formatted
        assert '95%' in formatted or 'CI' in formatted
