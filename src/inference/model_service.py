"""
Model service for loading and serving predictions.

This module handles model lifecycle management, prediction logic,
and SHAP-based explanations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import structlog

from src.features.pipeline import FeaturePipeline
from src.inference.schemas import (
    FeatureContribution,
    LoanApplication,
    PredictionExplanation,
    Recommendation,
    RiskPrediction,
    RiskTier,
)

logger = structlog.get_logger()


@dataclass
class ModelService:
    """
    Service for model prediction and explanation.

    Handles model loading, feature transformation, prediction,
    and SHAP-based explanations in a unified interface.
    """

    model_dir: str | Path
    config: dict[str, Any]
    model: Any = None
    pipeline: FeaturePipeline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    explainer: shap.TreeExplainer | None = None
    _loaded: bool = False

    def load(self, version: str = "latest") -> ModelService:
        """
        Load model and associated artifacts.

        Args:
            version: Model version to load (or 'latest')

        Returns:
            Self for method chaining
        """
        model_dir = Path(self.model_dir)
        version_dir = model_dir / version

        # Resolve 'latest' symlink
        if version == "latest" and version_dir.is_symlink():
            version_dir = version_dir.resolve()

        if not version_dir.exists():
            raise FileNotFoundError(f"Model version not found: {version_dir}")

        logger.info("loading_model", version_dir=str(version_dir))

        # Load model
        model_path = version_dir / "model.joblib"
        self.model = joblib.load(model_path)

        # Load pipeline
        pipeline_path = version_dir / "pipeline.json"
        self.pipeline = FeaturePipeline.load(pipeline_path, self.config)

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Initialize SHAP explainer if enabled
        if self.config.get("inference", {}).get("shap", {}).get("enabled", False):
            self._init_explainer()

        self._loaded = True
        logger.info(
            "model_loaded",
            version=self.metadata.get("model_version"),
            n_features=self.metadata.get("n_features"),
        )

        return self

    def _init_explainer(self) -> None:
        """Initialize SHAP TreeExplainer for model explanations."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("shap_explainer_initialized")
        except Exception as e:
            logger.warning("shap_explainer_init_failed", error=str(e))
            self.explainer = None

    def predict(
        self,
        application: LoanApplication,
        include_explanation: bool = True,
    ) -> RiskPrediction:
        """
        Generate risk prediction for a single loan application.

        Args:
            application: Loan application data
            include_explanation: Whether to include SHAP explanation

        Returns:
            RiskPrediction with score, tier, recommendation, and optional explanation
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Convert to dict and transform features
        app_dict = application.model_dump()
        features = self.pipeline.transform_single(app_dict)

        # Create DataFrame for prediction
        feature_names = self.pipeline.get_feature_names()
        X = pd.DataFrame([features])[feature_names]

        # Get prediction probability
        proba = self.model.predict_proba(X)[0, 1]

        # Determine risk tier
        risk_tier, recommendation = self._get_risk_tier(proba)

        # Generate explanation if requested
        explanation = None
        if include_explanation and self.explainer is not None:
            for feature in self.pipeline.derived_features:
                if not X[feature.get("name")].empty and feature.get("name") not in app_dict:
                    app_dict[feature.get("name")] = round(X[feature.get("name")].iloc[0], 4)
            explanation = self._generate_explanation(X, app_dict)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "prediction_complete",
            risk_score=round(proba, 4),
            risk_tier=risk_tier.value,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return RiskPrediction(
            risk_score=round(float(proba), 4),
            risk_tier=risk_tier,
            recommendation=recommendation,
            model_version=self.metadata.get("model_version", "unknown"),
            explanation=explanation,
        )

    def predict_batch(
        self,
        applications: list[LoanApplication],
        include_explanations: bool = False,
    ) -> list[RiskPrediction]:
        """
        Generate predictions for multiple applications.

        Optimized for batch processing with vectorized operations.

        Args:
            applications: List of loan applications
            include_explanations: Whether to include SHAP explanations

        Returns:
            List of RiskPrediction objects
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Convert all applications to DataFrame
        app_dicts = [app.model_dump() for app in applications]
        df = pd.DataFrame(app_dicts)

        # Transform features
        X = self.pipeline.transform(df)
        feature_names = self.pipeline.get_feature_names()
        X = X[feature_names]

        # Batch prediction
        probs = self.model.predict_proba(X)[:, 1]

        # Generate explanations if requested
        explanations: list[PredictionExplanation | None] = [None] * len(applications)
        if include_explanations and self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification

            for i, (app_dict, shap_vals) in enumerate(zip(app_dicts, shap_values, strict=True)):
                explanations[i] = self._format_explanation(shap_vals, feature_names, app_dict)

        # Build response
        predictions = []
        for i, prob in enumerate(probs):
            risk_tier, recommendation = self._get_risk_tier(prob)
            predictions.append(
                RiskPrediction(
                    risk_score=round(float(prob), 4),
                    risk_tier=risk_tier,
                    recommendation=recommendation,
                    model_version=self.metadata.get("model_version", "unknown"),
                    explanation=explanations[i],
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "batch_prediction_complete",
            n_samples=len(applications),
            elapsed_ms=round(elapsed_ms, 2),
            avg_ms=round(elapsed_ms / len(applications), 2),
        )

        return predictions

    def _get_risk_tier(self, proba: float) -> tuple[RiskTier, Recommendation]:
        """Map probability to risk tier and recommendation."""
        risk_tiers = self.config.get("inference", {}).get("risk_tiers", [])

        for tier in risk_tiers:
            if proba <= tier["max_score"]:
                risk_tier = RiskTier(tier["name"])
                recommendation = Recommendation(tier["recommendation"])
                return risk_tier, recommendation

        # Default to the highest risk if no tier matches
        return RiskTier.VERY_HIGH, Recommendation.DECLINE

    def _generate_explanation(
        self, X: pd.DataFrame, original_input: dict[str, Any]
    ) -> PredictionExplanation:
        """Generate SHAP-based explanation for a single prediction."""
        shap_values = self.explainer.shap_values(X)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class

        feature_names = self.pipeline.get_feature_names()
        return self._format_explanation(shap_values[0], feature_names, original_input)

    def _format_explanation(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        original_input: dict[str, Any],
    ) -> PredictionExplanation:
        """Format SHAP values into explanation schema."""
        max_features = (
            self.config.get("inference", {}).get("shap", {}).get("max_display_features", 10)
        )

        # Get base value (expected value)
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]  # Positive class
        else:
            base_value = 0.5

        # Sort features by absolute contribution
        feature_contributions = list(zip(feature_names, shap_values, strict=False))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Format top contributors
        top_contributors = []
        for name, contribution in feature_contributions[:max_features]:
            # Get original value if available
            original_value = original_input.get(name, "N/A")

            top_contributors.append(
                FeatureContribution(
                    feature=name,
                    value=original_value,
                    contribution=round(float(contribution), 4),
                )
            )

        return PredictionExplanation(
            base_value=round(float(base_value), 4),
            top_contributors=top_contributors,
        )

    def get_info(self) -> dict[str, Any]:
        """Get model metadata and info."""
        if not self._loaded:
            return {"status": "not_loaded"}

        return {
            "model_name": self.metadata.get("model_name"),
            "model_version": self.metadata.get("model_version"),
            "model_type": self.metadata.get("model_type"),
            "n_features": self.metadata.get("n_features"),
            "feature_names": self.metadata.get("feature_names"),
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics", {}),
        }

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def version(self) -> str | None:
        """Get loaded model version."""
        return self.metadata.get("model_version") if self._loaded else None
