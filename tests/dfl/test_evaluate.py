"""Tests for Phase 3: Evaluation metrics and ablation analysis.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import pytest
import torch
import numpy as np


class TestMetrics:
    """Tests for evaluation metrics computation."""

    def test_compute_auc_exists(self):
        """compute_auc function should exist."""
        from src.dfl.evaluate import compute_auc

        assert compute_auc is not None

    def test_compute_auc_returns_float(self):
        """compute_auc should return a float between 0 and 1."""
        from src.dfl.evaluate import compute_auc

        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]

        auc = compute_auc(y_true, y_pred)

        assert isinstance(auc, float)
        assert 0 <= auc <= 1

    def test_compute_auc_perfect_prediction(self):
        """Perfect predictions should give AUC = 1.0."""
        from src.dfl.evaluate import compute_auc

        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.9, 0.95]

        auc = compute_auc(y_true, y_pred)

        assert auc == 1.0

    def test_compute_auc_random_prediction(self):
        """Random predictions should give AUC around 0.5."""
        from src.dfl.evaluate import compute_auc

        np.random.seed(42)
        y_true = [0] * 50 + [1] * 50
        y_pred = list(np.random.rand(100))

        auc = compute_auc(y_true, y_pred)

        # Should be close to 0.5 for random predictions
        assert 0.3 <= auc <= 0.7

    def test_compute_f1_exists(self):
        """compute_f1 function should exist."""
        from src.dfl.evaluate import compute_f1

        assert compute_f1 is not None

    def test_compute_f1_returns_float(self):
        """compute_f1 should return a float between 0 and 1."""
        from src.dfl.evaluate import compute_f1

        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]

        f1 = compute_f1(y_true, y_pred)

        assert isinstance(f1, float)
        assert 0 <= f1 <= 1


class TestCrossValidationResults:
    """Tests for aggregating cross-validation results."""

    def test_aggregate_fold_results_exists(self):
        """aggregate_fold_results function should exist."""
        from src.dfl.evaluate import aggregate_fold_results

        assert aggregate_fold_results is not None

    def test_aggregate_fold_results_returns_dict(self):
        """aggregate_fold_results should return a dict with mean and std."""
        from src.dfl.evaluate import aggregate_fold_results

        fold_results = [
            {"test_match": "match_1", "predictions": [0.5, 0.7], "labels": [0, 1]},
            {"test_match": "match_2", "predictions": [0.3, 0.8], "labels": [0, 1]},
            {"test_match": "match_3", "predictions": [0.6, 0.9], "labels": [1, 1]},
        ]

        result = aggregate_fold_results(fold_results)

        assert isinstance(result, dict)
        assert "mean_auc" in result
        assert "std_auc" in result

    def test_aggregate_fold_results_computes_per_fold_auc(self):
        """aggregate_fold_results should compute AUC for each fold."""
        from src.dfl.evaluate import aggregate_fold_results

        fold_results = [
            {"test_match": "match_1", "predictions": [0.1, 0.9], "labels": [0, 1]},
            {"test_match": "match_2", "predictions": [0.2, 0.8], "labels": [0, 1]},
        ]

        result = aggregate_fold_results(fold_results)

        assert "fold_aucs" in result
        assert len(result["fold_aucs"]) == 2


class TestStatisticalTests:
    """Tests for statistical significance testing."""

    def test_paired_t_test_exists(self):
        """paired_t_test function should exist."""
        from src.dfl.evaluate import paired_t_test

        assert paired_t_test is not None

    def test_paired_t_test_returns_dict(self):
        """paired_t_test should return dict with t_stat and p_value."""
        from src.dfl.evaluate import paired_t_test

        scores_a = [0.5, 0.52, 0.48, 0.51, 0.49]
        scores_b = [0.6, 0.62, 0.58, 0.61, 0.59]

        result = paired_t_test(scores_a, scores_b)

        assert isinstance(result, dict)
        assert "t_stat" in result
        assert "p_value" in result

    def test_paired_t_test_significant_difference(self):
        """Large difference should give small p-value."""
        from src.dfl.evaluate import paired_t_test

        # Clearly different distributions
        scores_a = [0.5, 0.5, 0.5, 0.5, 0.5]
        scores_b = [0.9, 0.9, 0.9, 0.9, 0.9]

        result = paired_t_test(scores_a, scores_b)

        assert result["p_value"] < 0.05

    def test_paired_t_test_no_difference(self):
        """Identical scores should give high p-value."""
        from src.dfl.evaluate import paired_t_test

        scores_a = [0.5, 0.5, 0.5, 0.5, 0.5]
        scores_b = [0.5, 0.5, 0.5, 0.5, 0.5]

        result = paired_t_test(scores_a, scores_b)

        # p-value should be high (or nan for identical values)
        assert result["p_value"] > 0.05 or np.isnan(result["p_value"])


class TestAblationAnalysis:
    """Tests for analyzing ablation experiment results."""

    def test_analyze_ablation_results_exists(self):
        """analyze_ablation_results function should exist."""
        from src.dfl.evaluate import analyze_ablation_results

        assert analyze_ablation_results is not None

    def test_analyze_ablation_results_returns_dict(self):
        """analyze_ablation_results should return analysis dict."""
        from src.dfl.evaluate import analyze_ablation_results

        ablation_results = {
            "position_only": [
                {"test_match": "m1", "predictions": [0.5], "labels": [1]},
                {"test_match": "m2", "predictions": [0.5], "labels": [0]},
            ],
            "position_velocity": [
                {"test_match": "m1", "predictions": [0.7], "labels": [1]},
                {"test_match": "m2", "predictions": [0.3], "labels": [0]},
            ],
        }

        analysis = analyze_ablation_results(ablation_results)

        assert isinstance(analysis, dict)

    def test_analyze_ablation_results_has_key_metrics(self):
        """Analysis should include mean AUCs and delta."""
        from src.dfl.evaluate import analyze_ablation_results

        ablation_results = {
            "position_only": [
                {"test_match": "m1", "predictions": [0.1, 0.9], "labels": [0, 1]},
                {"test_match": "m2", "predictions": [0.2, 0.8], "labels": [0, 1]},
            ],
            "position_velocity": [
                {"test_match": "m1", "predictions": [0.1, 0.95], "labels": [0, 1]},
                {"test_match": "m2", "predictions": [0.15, 0.85], "labels": [0, 1]},
            ],
        }

        analysis = analyze_ablation_results(ablation_results)

        assert "position_only_mean_auc" in analysis
        assert "position_velocity_mean_auc" in analysis
        assert "delta_auc" in analysis

    def test_analyze_ablation_results_has_significance(self):
        """Analysis should include significance test results."""
        from src.dfl.evaluate import analyze_ablation_results

        ablation_results = {
            "position_only": [
                {"test_match": "m1", "predictions": [0.1, 0.9], "labels": [0, 1]},
                {"test_match": "m2", "predictions": [0.2, 0.8], "labels": [0, 1]},
                {"test_match": "m3", "predictions": [0.15, 0.85], "labels": [0, 1]},
            ],
            "position_velocity": [
                {"test_match": "m1", "predictions": [0.1, 0.95], "labels": [0, 1]},
                {"test_match": "m2", "predictions": [0.15, 0.9], "labels": [0, 1]},
                {"test_match": "m3", "predictions": [0.1, 0.88], "labels": [0, 1]},
            ],
        }

        analysis = analyze_ablation_results(ablation_results)

        assert "p_value" in analysis


class TestResultsFormatting:
    """Tests for formatting results for reporting."""

    def test_format_ablation_report_exists(self):
        """format_ablation_report function should exist."""
        from src.dfl.evaluate import format_ablation_report

        assert format_ablation_report is not None

    def test_format_ablation_report_returns_string(self):
        """format_ablation_report should return a formatted string."""
        from src.dfl.evaluate import format_ablation_report

        analysis = {
            "position_only_mean_auc": 0.50,
            "position_only_std_auc": 0.02,
            "position_velocity_mean_auc": 0.58,
            "position_velocity_std_auc": 0.03,
            "delta_auc": 0.08,
            "p_value": 0.02,
        }

        report = format_ablation_report(analysis)

        assert isinstance(report, str)
        assert "Position-only" in report or "position" in report.lower()
        assert "0.50" in report or "50" in report
