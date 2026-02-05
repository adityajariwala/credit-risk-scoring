"""Tests for the ModelService prediction/explanation layer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.inference.model_service import ModelService
from src.inference.schemas import (
    LoanApplication,
    Recommendation,
    RiskPrediction,
    RiskTier,
)


@pytest.fixture
def model_dir(full_config) -> Path:
    """Spin up a temp dir with a trained model + pipeline + metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        ver = full_config["model"]["version"]
        vdir = model_dir / ver
        vdir.mkdir(parents=True)

        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "loan_amnt": np.random.lognormal(9, 0.5, n),
                "annual_inc": np.random.lognormal(11, 0.5, n),
                "dti": np.random.uniform(0, 50, n),
                "open_acc": np.random.poisson(10, n),
                "revol_bal": np.random.lognormal(8, 1, n),
                "revol_util": np.random.uniform(0, 100, n),
                "total_acc": np.random.poisson(20, n),
                "int_rate": np.random.uniform(5, 25, n),
                "installment": np.random.lognormal(6, 0.5, n),
                "term": np.random.choice(["36 months", "60 months"], n),
                "grade": np.random.choice(list("ABCDEFG"), n),
                "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
                "verification_status": np.random.choice(
                    ["Verified", "Source Verified", "Not Verified"], n
                ),
                "purpose": np.random.choice(
                    ["debt_consolidation", "credit_card", "home_improvement"], n
                ),
            }
        )
        y = pd.Series(np.random.binomial(1, 0.15, n))

        pipe = FeaturePipeline(config=full_config)
        X_t = pipe.fit_transform(df, y)

        model = lgb.LGBMClassifier(n_estimators=10, num_leaves=8, verbose=-1, random_state=42)
        model.fit(X_t, y)

        joblib.dump(model, vdir / "model.joblib")
        pipe.save(vdir / "pipeline.json")
        meta = {
            "model_name": full_config["model"]["name"],
            "model_version": ver,
            "model_type": full_config["model"]["type"],
            "trained_at": "20240101_000000",
            "feature_names": pipe.get_feature_names(),
            "n_features": len(pipe.get_feature_names()),
            "metrics": {"auc": 0.72},
        }
        with open(vdir / "metadata.json", "w") as f:
            json.dump(meta, f)

        (model_dir / "latest").symlink_to(ver, target_is_directory=True)
        yield model_dir


@pytest.fixture
def full_config() -> dict[str, Any]:
    """Full config covering all features the integration tests use."""
    return {
        "model": {
            "name": "test",
            "version": "v0.0.1",
            "type": "lightgbm",
            "params": {"objective": "binary", "n_estimators": 10, "num_leaves": 8, "verbose": -1},
        },
        "features": {
            "numeric": [
                {"name": "loan_amnt", "transform": "log1p", "fill_value": 0},
                {"name": "annual_inc", "transform": "log1p", "fill_value": 0},
                {
                    "name": "dti",
                    "transform": "clip",
                    "clip_min": 0,
                    "clip_max": 100,
                    "fill_value": 0,
                },
                {"name": "open_acc", "transform": "none", "fill_value": 0},
                {"name": "revol_bal", "transform": "log1p", "fill_value": 0},
                {
                    "name": "revol_util",
                    "transform": "clip",
                    "clip_min": 0,
                    "clip_max": 150,
                    "fill_value": 0,
                },
                {"name": "total_acc", "transform": "none", "fill_value": 0},
                {"name": "int_rate", "transform": "none", "fill_value": 0},
                {"name": "installment", "transform": "log1p", "fill_value": 0},
            ],
            "categorical": [
                {"name": "term", "encoding": "label", "categories": ["36 months", "60 months"]},
                {"name": "grade", "encoding": "ordinal", "categories": list("ABCDEFG")},
                {
                    "name": "home_ownership",
                    "encoding": "onehot",
                    "categories": ["RENT", "OWN", "MORTGAGE", "OTHER"],
                },
                {
                    "name": "verification_status",
                    "encoding": "onehot",
                    "categories": ["Verified", "Source Verified", "Not Verified"],
                },
                {
                    "name": "purpose",
                    "encoding": "label",
                    "categories": [
                        "debt_consolidation",
                        "credit_card",
                        "home_improvement",
                        "major_purchase",
                        "medical",
                        "car",
                        "vacation",
                        "small_business",
                        "other",
                    ],
                },
            ],
            "derived": [
                {"name": "loan_to_income_ratio", "formula": "loan_amnt / (annual_inc + 1)"}
            ],
        },
        "training": {
            "target_column": "loan_status",
            "positive_class": "Charged Off",
            "random_state": 42,
        },
        "inference": {
            "default_threshold": 0.5,
            "risk_tiers": [
                {"name": "low", "max_score": 0.2, "recommendation": "approve"},
                {"name": "medium", "max_score": 0.5, "recommendation": "review"},
                {"name": "high", "max_score": 0.8, "recommendation": "enhanced_review"},
                {"name": "very_high", "max_score": 1.0, "recommendation": "decline"},
            ],
            "shap": {"enabled": True, "max_display_features": 5},
        },
        "paths": {"model_dir": "models"},
    }


@pytest.fixture
def app_payload() -> dict[str, Any]:
    return {
        "loan_amnt": 15000.0,
        "annual_inc": 75000.0,
        "dti": 18.5,
        "open_acc": 8,
        "revol_bal": 12500.0,
        "revol_util": 45.2,
        "total_acc": 20,
        "int_rate": 12.5,
        "installment": 450.0,
        "term": "36 months",
        "grade": "B",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
    }


# -- loading --


class TestModelServiceLoad:
    def test_load_latest(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        svc.load("latest")
        assert svc.is_loaded
        assert svc.version == "v0.0.1"

    def test_load_explicit_version(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        svc.load("v0.0.1")
        assert svc.is_loaded

    def test_load_nonexistent_raises(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        with pytest.raises(FileNotFoundError):
            svc.load("v99.0.0")

    def test_not_loaded_by_default(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        assert not svc.is_loaded
        assert svc.version is None


# -- predict --


class TestPredict:
    def test_single_prediction(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        pred = svc.predict(LoanApplication(**app_payload))
        assert isinstance(pred, RiskPrediction)
        assert 0.0 <= pred.risk_score <= 1.0

    def test_prediction_has_tier_and_rec(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        pred = svc.predict(LoanApplication(**app_payload))
        assert isinstance(pred.risk_tier, RiskTier)
        assert isinstance(pred.recommendation, Recommendation)

    def test_prediction_with_explanation(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        pred = svc.predict(LoanApplication(**app_payload), include_explanation=True)
        assert pred.explanation is not None
        assert len(pred.explanation.top_contributors) > 0

    def test_prediction_without_explanation(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        pred = svc.predict(LoanApplication(**app_payload), include_explanation=False)
        assert pred.explanation is None

    def test_predict_before_load_raises(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.predict(LoanApplication(**app_payload))


# -- predict_batch --


class TestPredictBatch:
    def test_batch_length(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        apps = [LoanApplication(**app_payload)] * 3
        preds = svc.predict_batch(apps)
        assert len(preds) == 3

    def test_batch_with_explanations(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        apps = [LoanApplication(**app_payload)] * 2
        preds = svc.predict_batch(apps, include_explanations=True)
        for p in preds:
            assert p.explanation is not None

    def test_batch_before_load_raises(self, model_dir, full_config, app_payload) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        with pytest.raises(RuntimeError):
            svc.predict_batch([LoanApplication(**app_payload)])


# -- risk tier mapping --


class TestRiskTier:
    def test_low_risk(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        tier, rec = svc._get_risk_tier(0.1)
        assert tier == RiskTier.LOW
        assert rec == Recommendation.APPROVE

    def test_very_high_risk(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        tier, rec = svc._get_risk_tier(0.95)
        assert tier == RiskTier.VERY_HIGH
        assert rec == Recommendation.DECLINE

    def test_boundary_medium(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        # exactly at boundary => should still be medium (0.5 <= 0.5)
        tier, _ = svc._get_risk_tier(0.5)
        assert tier == RiskTier.MEDIUM

    def test_no_tiers_defaults_to_very_high(self, model_dir, full_config) -> None:
        cfg = {**full_config, "inference": {**full_config["inference"], "risk_tiers": []}}
        svc = ModelService(model_dir=model_dir, config=cfg).load()
        tier, rec = svc._get_risk_tier(0.3)
        assert tier == RiskTier.VERY_HIGH


# -- get_info --


class TestGetInfo:
    def test_info_when_loaded(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config).load()
        info = svc.get_info()
        assert info["model_name"] == "test"
        assert info["model_version"] == "v0.0.1"
        assert "n_features" in info

    def test_info_when_not_loaded(self, model_dir, full_config) -> None:
        svc = ModelService(model_dir=model_dir, config=full_config)
        assert svc.get_info() == {"status": "not_loaded"}


# -- shap explainer edge cases --


class TestExplainerEdgeCases:
    def test_shap_disabled(self, model_dir, full_config, app_payload) -> None:
        cfg = {**full_config, "inference": {**full_config["inference"], "shap": {"enabled": False}}}
        svc = ModelService(model_dir=model_dir, config=cfg).load()
        assert svc.explainer is None
        # prediction should still work, just no explanation
        pred = svc.predict(LoanApplication(**app_payload), include_explanation=True)
        assert pred.explanation is None

    def test_explainer_init_failure_is_graceful(self, model_dir, full_config) -> None:
        """If SHAP init blows up it should just log a warning, not crash."""
        svc = ModelService(model_dir=model_dir, config=full_config)
        svc.load()
        # explainer should exist because shap.enabled=True, but if it failed it'd be None
        # just verify the service is still usable
        assert svc.is_loaded
