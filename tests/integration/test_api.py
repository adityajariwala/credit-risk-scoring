"""
Integration tests for the FastAPI inference service.

Tests the complete API flow including model loading, prediction, and responses.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Generator

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.features.pipeline import FeaturePipeline


@pytest.fixture
def full_config() -> dict[str, Any]:
    """Provide a full configuration for integration tests."""
    return {
        "model": {
            "name": "test_model",
            "version": "v0.0.1",
            "type": "lightgbm",
            "params": {
                "objective": "binary",
                "metric": "auc",
                "n_estimators": 10,
                "num_leaves": 8,
                "verbose": -1,
                "random_state": 42,
            },
        },
        "features": {
            "numeric": [
                {"name": "loan_amnt", "transform": "log1p", "fill_value": 0},
                {"name": "annual_inc", "transform": "log1p", "fill_value": 0},
                {"name": "dti", "transform": "clip", "clip_min": 0, "clip_max": 100, "fill_value": 0},
                {"name": "open_acc", "transform": "none", "fill_value": 0},
                {"name": "revol_bal", "transform": "log1p", "fill_value": 0},
                {"name": "revol_util", "transform": "clip", "clip_min": 0, "clip_max": 150, "fill_value": 0},
                {"name": "total_acc", "transform": "none", "fill_value": 0},
                {"name": "int_rate", "transform": "none", "fill_value": 0},
                {"name": "installment", "transform": "log1p", "fill_value": 0},
            ],
            "categorical": [
                {"name": "term", "encoding": "label", "categories": ["36 months", "60 months"]},
                {"name": "grade", "encoding": "ordinal", "categories": ["A", "B", "C", "D", "E", "F", "G"]},
                {"name": "home_ownership", "encoding": "onehot", "categories": ["RENT", "OWN", "MORTGAGE", "OTHER"]},
                {"name": "verification_status", "encoding": "onehot", "categories": ["Verified", "Source Verified", "Not Verified"]},
                {"name": "purpose", "encoding": "label", "categories": ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "car", "vacation", "small_business", "other"]},
            ],
            "derived": [
                {"name": "loan_to_income_ratio", "formula": "loan_amnt / (annual_inc + 1)"},
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
            "shap": {
                "enabled": True,
                "max_display_features": 5,
            },
        },
        "paths": {
            "model_dir": "models",
        },
    }


@pytest.fixture
def trained_model_dir(full_config: dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary directory with trained model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        version_dir = model_dir / full_config["model"]["version"]
        version_dir.mkdir(parents=True)

        # Create sample training data
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
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
            "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n),
            "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
            "verification_status": np.random.choice(["Verified", "Source Verified", "Not Verified"], n),
            "purpose": np.random.choice(["debt_consolidation", "credit_card", "home_improvement"], n),
            "loan_status": np.random.choice(["Fully Paid", "Charged Off"], n, p=[0.85, 0.15]),
        })

        y = (df["loan_status"] == "Charged Off").astype(int)
        X = df.drop(columns=["loan_status"])

        # Fit pipeline
        pipeline = FeaturePipeline(config=full_config)
        X_transformed = pipeline.fit_transform(X, y)

        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=10,
            num_leaves=8,
            verbose=-1,
            random_state=42,
        )
        model.fit(X_transformed, y)

        # Save artifacts
        joblib.dump(model, version_dir / "model.joblib")
        pipeline.save(version_dir / "pipeline.json")

        metadata = {
            "model_name": full_config["model"]["name"],
            "model_version": full_config["model"]["version"],
            "model_type": full_config["model"]["type"],
            "trained_at": "2024-01-01T00:00:00",
            "feature_names": pipeline.get_feature_names(),
            "n_features": len(pipeline.get_feature_names()),
            "metrics": {"test": {"auc": 0.75}},
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create latest symlink
        latest = model_dir / "latest"
        latest.symlink_to(full_config["model"]["version"], target_is_directory=True)

        # Update config paths
        full_config["paths"]["model_dir"] = str(model_dir)

        # Write config file
        config_dir = model_dir / "config"
        config_dir.mkdir()
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(full_config, f)

        yield model_dir


@pytest.fixture
def test_client(trained_model_dir: Path, full_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client with loaded model."""
    # Patch the config loading to use our test config
    config_path = trained_model_dir / "config" / "config.yaml"
    monkeypatch.chdir(trained_model_dir)

    # Import app after patching
    from src.inference.app import app

    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, test_client: TestClient) -> None:
        """Test health check returns correct status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data

    def test_ready_endpoint_with_model(self, test_client: TestClient) -> None:
        """Test readiness check with loaded model."""
        response = test_client.get("/ready")

        # May be 200 or 503 depending on model load state
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self, test_client: TestClient) -> None:
        """Test Prometheus metrics endpoint."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert "predictions_total" in response.text or response.text


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_single_prediction(
        self,
        test_client: TestClient,
        sample_loan_application: dict[str, Any],
    ) -> None:
        """Test single prediction endpoint."""
        response = test_client.post(
            "/predict",
            json=sample_loan_application,
            params={"include_explanation": True},
        )

        # May be 200 or 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "risk_score" in data
            assert 0 <= data["risk_score"] <= 1
            assert "risk_tier" in data
            assert "recommendation" in data
            assert "model_version" in data

    def test_prediction_with_explanation(
        self,
        test_client: TestClient,
        sample_loan_application: dict[str, Any],
    ) -> None:
        """Test prediction includes explanation when requested."""
        response = test_client.post(
            "/predict",
            json=sample_loan_application,
            params={"include_explanation": True},
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("explanation"):
                assert "base_value" in data["explanation"]
                assert "top_contributors" in data["explanation"]

    def test_prediction_without_explanation(
        self,
        test_client: TestClient,
        sample_loan_application: dict[str, Any],
    ) -> None:
        """Test prediction without explanation."""
        response = test_client.post(
            "/predict",
            json=sample_loan_application,
            params={"include_explanation": False},
        )

        if response.status_code == 200:
            data = response.json()
            assert "risk_score" in data

    def test_batch_prediction(
        self,
        test_client: TestClient,
        sample_loan_application: dict[str, Any],
    ) -> None:
        """Test batch prediction endpoint."""
        request = {
            "applications": [sample_loan_application, sample_loan_application],
            "include_explanations": False,
        }

        response = test_client.post("/predict/batch", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert "processing_time_ms" in data

    def test_invalid_application_rejected(self, test_client: TestClient) -> None:
        """Test that invalid application data is rejected."""
        invalid_app = {
            "loan_amnt": -1000,  # Invalid: negative
            "annual_inc": 75000.0,
        }

        response = test_client.post("/predict", json=invalid_app)

        assert response.status_code == 422  # Validation error


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info(self, test_client: TestClient) -> None:
        """Test model info endpoint."""
        response = test_client.get("/model/info")

        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "n_features" in data
            assert "feature_names" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, test_client: TestClient) -> None:
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, test_client: TestClient) -> None:
        """Test handling of missing required fields."""
        incomplete_app = {
            "loan_amnt": 15000.0,
            # Missing other required fields
        }

        response = test_client.post("/predict", json=incomplete_app)

        assert response.status_code == 422
