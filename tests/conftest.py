"""
Pytest fixtures and configuration for test suite.

This module provides shared fixtures for testing the credit risk scoring system.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.pipeline import FeaturePipeline


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Provide a minimal configuration for testing."""
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
                {
                    "name": "dti",
                    "transform": "clip",
                    "clip_min": 0,
                    "clip_max": 100,
                    "fill_value": 0,
                },
            ],
            "categorical": [
                {
                    "name": "grade",
                    "encoding": "ordinal",
                    "categories": ["A", "B", "C", "D", "E", "F", "G"],
                },
                {
                    "name": "home_ownership",
                    "encoding": "onehot",
                    "categories": ["RENT", "OWN", "MORTGAGE", "OTHER"],
                },
            ],
            "derived": [
                {"name": "loan_to_income_ratio", "formula": "loan_amnt / (annual_inc + 1)"},
            ],
        },
        "training": {
            "target_column": "loan_status",
            "positive_class": "Charged Off",
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "cv_folds": 3,
            "class_weight": "balanced",
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
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "loan_amnt": np.random.lognormal(9, 0.5, n).clip(1000, 40000),
            "annual_inc": np.random.lognormal(11, 0.5, n).clip(20000, 200000),
            "dti": np.random.uniform(0, 50, n),
            "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n),
            "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
            "loan_status": np.random.choice(["Fully Paid", "Charged Off"], n, p=[0.85, 0.15]),
        }
    )


@pytest.fixture
def sample_target(sample_dataframe: pd.DataFrame) -> pd.Series:
    """Create target variable from sample DataFrame."""
    return (sample_dataframe["loan_status"] == "Charged Off").astype(int)


@pytest.fixture
def sample_loan_application() -> dict[str, Any]:
    """Provide a sample loan application for testing."""
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


@pytest.fixture
def fitted_pipeline(
    sample_config: dict[str, Any],
    sample_dataframe: pd.DataFrame,
    sample_target: pd.Series,
) -> FeaturePipeline:
    """Create and fit a feature pipeline for testing."""
    pipeline = FeaturePipeline(config=sample_config)
    pipeline.fit(sample_dataframe, sample_target)
    return pipeline


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_file(temp_dir: Path, sample_config: dict[str, Any]) -> Path:
    """Create a temporary config file for testing."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def saved_pipeline(fitted_pipeline: FeaturePipeline, temp_dir: Path) -> Path:
    """Save pipeline and return the path."""
    pipeline_path = temp_dir / "pipeline.json"
    fitted_pipeline.save(pipeline_path)
    return pipeline_path
