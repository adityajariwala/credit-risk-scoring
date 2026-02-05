"""Tests for training pipeline functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.training.train import (
    create_sample_data,
    cross_validate,
    load_config,
    load_data,
    save_artifacts,
    train_model,
)


@pytest.fixture
def tiny_config(sample_config: dict[str, Any]) -> dict[str, Any]:
    """Config tweaked so tests run fast."""
    cfg = sample_config.copy()
    cfg["model"] = {
        **cfg["model"],
        "params": {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 5,
            "num_leaves": 4,
            "verbose": -1,
            "random_state": 42,
        },
    }
    cfg["training"] = {**cfg["training"], "cv_folds": 2}
    return cfg


@pytest.fixture
def training_data(
    sample_config: dict[str, Any],
    sample_dataframe: pd.DataFrame,
    sample_target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    split = int(len(sample_dataframe) * 0.8)
    pipe = FeaturePipeline(config=sample_config)
    X_train = pipe.fit_transform(sample_dataframe.iloc[:split], sample_target.iloc[:split])
    X_val = pipe.transform(sample_dataframe.iloc[split:])
    return X_train, sample_target.iloc[:split], X_val, sample_target.iloc[split:]


# -- load_config --


class TestLoadConfig:
    def test_loads_yaml(self, config_file: Path) -> None:
        cfg = load_config(config_file)
        assert "model" in cfg and "training" in cfg

    def test_nonexistent_raises(self, temp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(temp_dir / "nope.yaml")


# -- load_data --


class TestLoadData:
    def test_returns_features_and_target(
        self,
        sample_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
        temp_dir: Path,
    ) -> None:
        csv = temp_dir / "data.csv"
        sample_dataframe.to_csv(csv, index=False)
        X, y = load_data(csv, sample_config)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert set(y.unique()).issubset({0, 1})

    def test_target_column_dropped(self, sample_config: dict[str, Any], temp_dir: Path) -> None:
        df = pd.DataFrame({"loan_amnt": [1000, 2000], "loan_status": ["Fully Paid", "Charged Off"]})
        df.to_csv(temp_dir / "d.csv", index=False)
        X, _ = load_data(temp_dir / "d.csv", sample_config)
        assert "loan_status" not in X.columns

    def test_id_columns_dropped(self, sample_config: dict[str, Any], temp_dir: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "member_id": [10, 20],
                "loan_amnt": [5000, 10000],
                "loan_status": ["Fully Paid", "Charged Off"],
            }
        )
        df.to_csv(temp_dir / "d.csv", index=False)
        X, _ = load_data(temp_dir / "d.csv", sample_config)
        assert "id" not in X.columns and "member_id" not in X.columns


# -- create_sample_data --


class TestCreateSampleData:
    def test_creates_csv_with_correct_rows(self, temp_dir: Path) -> None:
        out = temp_dir / "sample.csv"
        create_sample_data(out, n_samples=150)
        assert out.exists()
        assert len(pd.read_csv(out)) == 150

    def test_expected_columns(self, temp_dir: Path) -> None:
        out = temp_dir / "s.csv"
        create_sample_data(out, n_samples=50)
        cols = set(pd.read_csv(out).columns)
        for c in ["loan_amnt", "annual_inc", "dti", "grade", "loan_status", "purpose"]:
            assert c in cols

    def test_loan_status_values(self, temp_dir: Path) -> None:
        out = temp_dir / "s.csv"
        create_sample_data(out, n_samples=500)
        vals = set(pd.read_csv(out)["loan_status"].unique())
        assert vals.issubset({"Fully Paid", "Charged Off"})

    def test_creates_parent_dirs(self, temp_dir: Path) -> None:
        out = temp_dir / "deep" / "nested" / "data.csv"
        create_sample_data(out, n_samples=10)
        assert out.exists()

    def test_deterministic(self, temp_dir: Path) -> None:
        """Uses seed(42) internally so two calls should match."""
        a, b = temp_dir / "a.csv", temp_dir / "b.csv"
        create_sample_data(a, n_samples=50)
        create_sample_data(b, n_samples=50)
        pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))


# -- train_model --


class TestTrainModel:
    def test_returns_lgbm(self, tiny_config: dict[str, Any], training_data) -> None:
        X_tr, y_tr, X_v, y_v = training_data
        model = train_model(X_tr, y_tr, X_v, y_v, tiny_config)
        assert isinstance(model, lgb.LGBMClassifier)

    def test_predict_proba_shape(self, tiny_config: dict[str, Any], training_data) -> None:
        X_tr, y_tr, X_v, y_v = training_data
        model = train_model(X_tr, y_tr, X_v, y_v, tiny_config)
        probs = model.predict_proba(X_v)
        assert probs.shape == (len(X_v), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_works_without_class_weight(self, tiny_config: dict[str, Any], training_data) -> None:
        cfg = {
            **tiny_config,
            "training": {k: v for k, v in tiny_config["training"].items() if k != "class_weight"},
        }
        X_tr, y_tr, X_v, y_v = training_data
        assert train_model(X_tr, y_tr, X_v, y_v, cfg) is not None

    def test_early_stopping_rounds_popped(self, tiny_config: dict[str, Any], training_data) -> None:
        cfg = {**tiny_config}
        cfg["model"] = {
            **cfg["model"],
            "params": {**cfg["model"]["params"], "early_stopping_rounds": 3},
        }
        X_tr, y_tr, X_v, y_v = training_data
        # shouldn't blow up
        assert isinstance(train_model(X_tr, y_tr, X_v, y_v, cfg), lgb.LGBMClassifier)

    def test_predictions_in_unit_interval(self, tiny_config: dict[str, Any], training_data) -> None:
        X_tr, y_tr, X_v, y_v = training_data
        model = train_model(X_tr, y_tr, X_v, y_v, tiny_config)
        preds = model.predict_proba(X_v)[:, 1]
        assert preds.min() >= 0.0 and preds.max() <= 1.0


# -- cross_validate --


class TestCrossValidate:
    def test_returns_expected_keys(self, tiny_config: dict[str, Any]) -> None:
        np.random.seed(99)
        X = pd.DataFrame(np.random.randn(120, 4), columns=list("abcd"))
        y = pd.Series([0] * 100 + [1] * 20)
        results = cross_validate(X, y, tiny_config)
        for k in ("auc", "precision", "recall", "f1"):
            assert k in results

    def test_fold_count(self, tiny_config: dict[str, Any]) -> None:
        np.random.seed(7)
        X = pd.DataFrame(np.random.randn(100, 3), columns=list("xyz"))
        y = pd.Series([0] * 85 + [1] * 15)
        results = cross_validate(X, y, tiny_config)
        for vals in results.values():
            assert len(vals) == tiny_config["training"]["cv_folds"]

    def test_metrics_in_range(self, tiny_config: dict[str, Any]) -> None:
        np.random.seed(42)
        # give it some signal
        X = pd.DataFrame(
            {
                "s": np.concatenate([np.random.randn(100) - 0.5, np.random.randn(20) + 1.0]),
                "n": np.random.randn(120),
            }
        )
        y = pd.Series([0] * 100 + [1] * 20)
        for vals in cross_validate(X, y, tiny_config).values():
            for v in vals:
                assert 0.0 <= v <= 1.0


# -- save_artifacts --


class TestSaveArtifacts:
    @pytest.fixture
    def _trained(self, tiny_config, sample_dataframe, sample_target):
        split = int(len(sample_dataframe) * 0.8)
        pipe = FeaturePipeline(config=tiny_config)
        X_tr = pipe.fit_transform(sample_dataframe.iloc[:split], sample_target.iloc[:split])
        X_v = pipe.transform(sample_dataframe.iloc[split:])
        model = train_model(
            X_tr, sample_target.iloc[:split], X_v, sample_target.iloc[split:], tiny_config
        )
        return model, pipe, tiny_config

    def test_creates_version_dir(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        save_artifacts(model, pipe, cfg, {"auc": 0.75}, temp_dir)
        assert (temp_dir / cfg["model"]["version"]).is_dir()

    def test_model_loadable(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        result = save_artifacts(model, pipe, cfg, {}, temp_dir)
        loaded = joblib.load(result["model"])
        assert isinstance(loaded, lgb.LGBMClassifier)

    def test_pipeline_saved_as_json(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        result = save_artifacts(model, pipe, cfg, {}, temp_dir)
        assert Path(result["pipeline"]).suffix == ".json"

    def test_metadata_contents(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        result = save_artifacts(model, pipe, cfg, {"auc": 0.85}, temp_dir)
        with open(result["metadata"]) as f:
            meta = json.load(f)
        assert meta["model_name"] == cfg["model"]["name"]
        assert meta["model_version"] == cfg["model"]["version"]
        assert "trained_at" in meta
        assert meta["n_features"] == len(pipe.get_feature_names())

    def test_latest_symlink(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        save_artifacts(model, pipe, cfg, {}, temp_dir)
        latest = temp_dir / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == (temp_dir / cfg["model"]["version"]).resolve()

    def test_symlink_updated_on_new_version(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        save_artifacts(model, pipe, cfg, {}, temp_dir)
        cfg2 = {**cfg, "model": {**cfg["model"], "version": "v0.0.2"}}
        save_artifacts(model, pipe, cfg2, {}, temp_dir)
        assert (temp_dir / "latest").resolve() == (temp_dir / "v0.0.2").resolve()

    def test_returns_all_paths(self, _trained, temp_dir) -> None:
        model, pipe, cfg = _trained
        result = save_artifacts(model, pipe, cfg, {}, temp_dir)
        for key in ("model", "pipeline", "metadata", "version_dir"):
            assert key in result

    def test_end_to_end_save_load_predict(self, _trained, temp_dir) -> None:
        """Save artifacts, reload them, make a prediction."""
        model, pipe, cfg = _trained
        result = save_artifacts(model, pipe, cfg, {}, temp_dir)

        loaded_model = joblib.load(result["model"])
        loaded_pipe = FeaturePipeline.load(result["pipeline"])

        row = {
            "loan_amnt": 10000.0,
            "annual_inc": 60000.0,
            "dti": 15.0,
            "grade": "C",
            "home_ownership": "RENT",
        }
        feats = loaded_pipe.transform_single(row)
        proba = loaded_model.predict_proba(pd.DataFrame([feats]))[0, 1]
        assert 0.0 <= proba <= 1.0


# -- round-trip: create -> load --


class TestDataRoundTrip:
    def test_create_then_load(self, sample_config: dict[str, Any], temp_dir: Path) -> None:
        csv = temp_dir / "rt.csv"
        create_sample_data(csv, n_samples=100)
        X, y = load_data(csv, sample_config)
        assert len(X) == 100 and "loan_status" not in X.columns

    def test_default_rate_is_sane(self, sample_config: dict[str, Any], temp_dir: Path) -> None:
        csv = temp_dir / "dr.csv"
        create_sample_data(csv, n_samples=5000)
        _, y = load_data(csv, sample_config)
        rate = y.mean()
        assert 0.05 < rate < 0.40, f"default rate {rate:.2%} seems off"
