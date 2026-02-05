"""
Evaluation metrics for credit risk scoring model.

This module provides comprehensive metrics computation for binary classification,
including threshold-based business metrics and model performance indicators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute comprehensive model evaluation metrics.

    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Check if we have both classes for AUC computation
    n_classes = len(np.unique(y_true))

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC metrics require both classes present
    if n_classes > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    else:
        # Cannot compute AUC with single class
        metrics["auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # Confusion matrix elements - handle single-class case
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    # Additional business metrics
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    return metrics


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    risk_tiers: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Compute metrics at different risk tier thresholds.

    This helps in understanding model performance across business-defined
    risk categories.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        risk_tiers: List of risk tier configurations from config

    Returns:
        Dictionary mapping tier names to their metrics
    """
    results = {}
    prev_threshold = 0.0

    for tier in risk_tiers:
        tier_name = tier["name"]
        threshold = tier["max_score"]

        # Samples in this tier
        mask = (y_pred_proba > prev_threshold) & (y_pred_proba <= threshold)
        tier_samples = mask.sum()

        if tier_samples > 0:
            tier_default_rate = y_true[mask].mean()
            tier_volume_pct = tier_samples / len(y_true)
        else:
            tier_default_rate = 0.0
            tier_volume_pct = 0.0

        results[tier_name] = {
            "samples": int(tier_samples),
            "volume_pct": round(tier_volume_pct, 4),
            "default_rate": round(tier_default_rate, 4),
            "recommendation": tier["recommendation"],
        }

        prev_threshold = threshold

    return results


def compute_lift_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """
    Compute lift curve data for model evaluation.

    Lift shows how much better the model performs compared to random selection.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for lift calculation

    Returns:
        Dictionary with percentiles, lift values, and cumulative captures
    """
    # Sort by predicted probability descending
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]

    n_samples = len(y_true)
    baseline_rate = y_true.mean()

    percentiles = []
    lift_values = []
    cumulative_captures = []

    for i in range(1, n_bins + 1):
        percentile = i / n_bins
        cutoff = int(percentile * n_samples)

        top_k_rate = y_true_sorted[:cutoff].mean()
        lift = top_k_rate / baseline_rate if baseline_rate > 0 else 0
        cumulative_capture = y_true_sorted[:cutoff].sum() / y_true.sum() if y_true.sum() > 0 else 0

        percentiles.append(percentile)
        lift_values.append(round(lift, 4))
        cumulative_captures.append(round(cumulative_capture, 4))

    return {
        "percentiles": percentiles,
        "lift": lift_values,
        "cumulative_capture": cumulative_captures,
    }


def compute_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    """
    Compute Kolmogorov-Smirnov statistic.

    KS measures the maximum separation between cumulative distribution
    of positives and negatives. Common in credit risk models.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary with KS statistic and threshold where it occurs
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # KS is max difference between TPR and FPR
    ks_values = tpr - fpr
    ks_idx = np.argmax(ks_values)

    return {
        "ks_statistic": round(ks_values[ks_idx], 4),
        "ks_threshold": round(thresholds[ks_idx], 4) if len(thresholds) > ks_idx else 0.5,
    }


def compute_gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Gini coefficient (2 * AUC - 1).

    Gini is commonly used in credit risk to measure model discriminatory power.
    Range: 0 (random) to 1 (perfect).

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Gini coefficient
    """
    auc_score = roc_auc_score(y_true, y_pred_proba)
    return round(2 * auc_score - 1, 4)


def compute_population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) for drift detection.

    PSI measures how much a distribution has shifted between two samples.
    Used to detect prediction drift in production.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change, investigate

    Args:
        expected: Reference distribution (e.g., training predictions)
        actual: Current distribution (e.g., production predictions)
        n_bins: Number of bins

    Returns:
        PSI value
    """
    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-10, 1)
    actual_pct = np.clip(actual_pct, 1e-10, 1)

    # Compute PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return round(psi, 4)
