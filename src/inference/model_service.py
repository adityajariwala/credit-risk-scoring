"""
Model loading, prediction, and SHAP explanation logic.

Sits between the FastAPI routes and the raw LightGBM model — handles
feature transformation, probability thresholding, and explanation formatting.
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
    """Wraps a trained model + pipeline for serving."""

    model_dir: str | Path
    config: dict[str, Any]
    model: Any = None
    pipeline: FeaturePipeline | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    explainer: shap.TreeExplainer | None = None
    _loaded: bool = False

    def load(self, version: str = "latest") -> ModelService:
        """Load model, pipeline, and metadata from disk. Returns self for chaining."""
        model_dir = Path(self.model_dir)
        version_dir = model_dir / version

        if version == "latest" and version_dir.is_symlink():
            version_dir = version_dir.resolve()

        if not version_dir.exists():
            raise FileNotFoundError(f"Model version not found: {version_dir}")

        logger.info("loading_model", version_dir=str(version_dir))

        self.model = joblib.load(version_dir / "model.joblib")
        self.pipeline = FeaturePipeline.load(version_dir / "pipeline.json", self.config)

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

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
        """Score a single application. Optionally attaches SHAP explanation."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        app_dict = application.model_dump()
        features = self.pipeline.transform_single(app_dict)
        feature_names = self.pipeline.get_feature_names()
        X = pd.DataFrame([features])[feature_names]

        proba = self.model.predict_proba(X)[0, 1]
        risk_tier, recommendation = self._get_risk_tier(proba)

        explanation = None
        if include_explanation and self.explainer is not None:
            for feature in self.pipeline.derived_features:
                feature_name = feature.get("name")
                if (
                    isinstance(feature_name, str)
                    and not X[feature_name].empty
                    and feature_name not in app_dict
                ):
                    app_dict[feature_name] = round(X[feature_name].iloc[0], 4)
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
        """Vectorized scoring for multiple applications at once."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        app_dicts = [app.model_dump() for app in applications]
        df = pd.DataFrame(app_dicts)
        X = self.pipeline.transform(df)
        feature_names = self.pipeline.get_feature_names()
        X = X[feature_names]

        probs = self.model.predict_proba(X)[:, 1]

        explanations: list[PredictionExplanation | None] = [None] * len(applications)
        if include_explanations and self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # binary clf returns [neg_class, pos_class]

            for i, (app_dict, shap_vals) in enumerate(zip(app_dicts, shap_values, strict=True)):
                explanations[i] = self._format_explanation(shap_vals, feature_names, app_dict)

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
        """Walk the tier list and return the first match."""
        risk_tiers = self.config.get("inference", {}).get("risk_tiers", [])

        for tier in risk_tiers:
            if proba <= tier["max_score"]:
                risk_tier = RiskTier(tier["name"])
                recommendation = Recommendation(tier["recommendation"])
                return risk_tier, recommendation

        return RiskTier.VERY_HIGH, Recommendation.DECLINE  # fallback

    def _generate_explanation(
        self, X: pd.DataFrame, original_input: dict[str, Any]
    ) -> PredictionExplanation:
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        feature_names = self.pipeline.get_feature_names()
        return self._format_explanation(shap_values[0], feature_names, original_input)

    def _format_explanation(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        original_input: dict[str, Any],
    ) -> PredictionExplanation:
        """Package raw SHAP values into the API-friendly schema."""
        max_features = (
            self.config.get("inference", {}).get("shap", {}).get("max_display_features", 10)
        )

        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        else:
            base_value = 0.5

        feature_contributions = list(zip(feature_names, shap_values, strict=False))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_contributors = []
        for name, contribution in feature_contributions[:max_features]:
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
        return self._loaded

    @property
    def version(self) -> str | None:
        return self.metadata.get("model_version") if self._loaded else None
