"""
Unit tests for evaluation metrics.

Tests metric computation accuracy and edge cases.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_gini_coefficient,
    compute_ks_statistic,
    compute_lift_curve,
    compute_population_stability_index,
    compute_threshold_metrics,
    evaluate_model,
)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.1, 0.9, 1.0])

        metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

        assert metrics["auc"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self) -> None:
        """Test metrics with random predictions should have ~0.5 AUC."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_pred_proba = np.random.random(n)

        metrics = evaluate_model(y_true, y_pred_proba)

        # Random should have AUC around 0.5
        assert 0.4 < metrics["auc"] < 0.6

    def test_confusion_matrix_elements(self) -> None:
        """Test confusion matrix elements are correct."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.6, 0.4, 0.8, 0.9])

        metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

        # TN=1, FP=1, FN=1, TP=2
        assert metrics["true_negative"] == 1
        assert metrics["false_positive"] == 1
        assert metrics["false_negative"] == 1
        assert metrics["true_positive"] == 2

    def test_all_same_class(self) -> None:
        """Test handling when all samples are same class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

        # Should not raise errors
        metrics = evaluate_model(y_true, y_pred_proba)

        assert "accuracy" in metrics


class TestThresholdMetrics:
    """Tests for compute_threshold_metrics function."""

    def test_tier_distribution(self) -> None:
        """Test that samples are correctly distributed across tiers."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95])

        risk_tiers = [
            {"name": "low", "max_score": 0.2, "recommendation": "approve"},
            {"name": "medium", "max_score": 0.5, "recommendation": "review"},
            {"name": "high", "max_score": 0.8, "recommendation": "enhanced_review"},
            {"name": "very_high", "max_score": 1.0, "recommendation": "decline"},
        ]

        result = compute_threshold_metrics(y_true, y_pred_proba, risk_tiers)

        assert result["low"]["samples"] == 2  # 0.1, 0.15
        assert result["medium"]["samples"] == 3  # 0.3, 0.4, 0.5
        assert result["high"]["samples"] == 2  # 0.6, 0.7
        assert result["very_high"]["samples"] == 3  # 0.85, 0.9, 0.95

    def test_tier_default_rates(self) -> None:
        """Test that default rates increase with risk tiers."""
        # Create data where higher scores correspond to higher default rates
        n = 1000
        np.random.seed(42)

        y_pred_proba = np.random.random(n)
        y_true = (np.random.random(n) < y_pred_proba).astype(int)

        risk_tiers = [
            {"name": "low", "max_score": 0.25, "recommendation": "approve"},
            {"name": "medium", "max_score": 0.5, "recommendation": "review"},
            {"name": "high", "max_score": 0.75, "recommendation": "enhanced_review"},
            {"name": "very_high", "max_score": 1.0, "recommendation": "decline"},
        ]

        result = compute_threshold_metrics(y_true, y_pred_proba, risk_tiers)

        # Default rates should generally increase
        assert result["low"]["default_rate"] < result["very_high"]["default_rate"]


class TestLiftCurve:
    """Tests for compute_lift_curve function."""

    def test_lift_values(self) -> None:
        """Test lift curve computation."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

        result = compute_lift_curve(y_true, y_pred_proba, n_bins=5)

        assert len(result["percentiles"]) == 5
        assert len(result["lift"]) == 5
        assert len(result["cumulative_capture"]) == 5

        # Top 20% should have high lift (all positives are ranked at top)
        assert result["lift"][0] > 1

    def test_perfect_model_lift(self) -> None:
        """Test lift curve with perfect model."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred_proba = np.array([1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        result = compute_lift_curve(y_true, y_pred_proba, n_bins=5)

        # Top 20% should capture 100% of positives
        assert result["cumulative_capture"][0] == 1.0


class TestKSStatistic:
    """Tests for compute_ks_statistic function."""

    def test_ks_perfect_separation(self) -> None:
        """Test KS with perfect separation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        result = compute_ks_statistic(y_true, y_pred_proba)

        # Perfect separation should have KS = 1.0
        assert result["ks_statistic"] == 1.0

    def test_ks_random_model(self) -> None:
        """Test KS with random predictions."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_pred_proba = np.random.random(n)

        result = compute_ks_statistic(y_true, y_pred_proba)

        # Random model should have KS close to 0
        assert result["ks_statistic"] < 0.1


class TestGiniCoefficient:
    """Tests for compute_gini_coefficient function."""

    def test_gini_perfect_model(self) -> None:
        """Test Gini with perfect model."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.1, 0.9, 1.0])

        gini = compute_gini_coefficient(y_true, y_pred_proba)

        # Perfect model: AUC=1.0, Gini=1.0
        assert gini == 1.0

    def test_gini_random_model(self) -> None:
        """Test Gini with random model."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_pred_proba = np.random.random(n)

        gini = compute_gini_coefficient(y_true, y_pred_proba)

        # Random model: AUC~0.5, Gini~0
        assert abs(gini) < 0.1


class TestPSI:
    """Tests for compute_population_stability_index function."""

    def test_psi_same_distribution(self) -> None:
        """Test PSI when distributions are identical."""
        np.random.seed(42)
        expected = np.random.normal(0.5, 0.1, 1000)
        actual = np.random.normal(0.5, 0.1, 1000)

        psi = compute_population_stability_index(expected, actual)

        # Same distribution should have low PSI
        assert psi < 0.1

    def test_psi_shifted_distribution(self) -> None:
        """Test PSI when distribution has shifted."""
        np.random.seed(42)
        expected = np.random.normal(0.3, 0.1, 1000)
        actual = np.random.normal(0.7, 0.1, 1000)  # Significant shift

        psi = compute_population_stability_index(expected, actual)

        # Large shift should have high PSI
        assert psi > 0.2

    def test_psi_handles_empty_bins(self) -> None:
        """Test PSI handles empty bins gracefully."""
        expected = np.array([0.1, 0.2, 0.3] * 100)
        actual = np.array([0.8, 0.9, 0.95] * 100)  # Completely different range

        # Should not raise errors
        psi = compute_population_stability_index(expected, actual)

        assert psi > 0  # Should indicate significant shift
