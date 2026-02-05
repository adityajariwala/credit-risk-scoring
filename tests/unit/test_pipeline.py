"""
Unit tests for the feature pipeline.

Tests pipeline fitting, transformation, serialization, and parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline


class TestFeaturePipeline:
    """Tests for FeaturePipeline class."""

    def test_fit_creates_transforms(
        self,
        sample_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test that fitting creates all expected transforms."""
        pipeline = FeaturePipeline(config=sample_config)
        pipeline.fit(sample_dataframe, sample_target)

        # Check numeric transforms were created
        for feat in sample_config["features"]["numeric"]:
            assert feat["name"] in pipeline.numeric_transforms

        # Check categorical transforms were created
        for feat in sample_config["features"]["categorical"]:
            assert feat["name"] in pipeline.categorical_transforms

    def test_transform_output_shape(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test that transform produces expected output shape."""
        result = fitted_pipeline.transform(sample_dataframe)

        assert len(result) == len(sample_dataframe)
        assert len(result.columns) == len(fitted_pipeline.get_feature_names())

    def test_transform_no_nan(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test that transformed output has no NaN values."""
        result = fitted_pipeline.transform(sample_dataframe)

        assert not result.isna().any().any()

    def test_fit_transform_equivalence(
        self,
        sample_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test that fit_transform equals fit then transform."""
        pipeline1 = FeaturePipeline(config=sample_config)
        result1 = pipeline1.fit_transform(sample_dataframe, sample_target)

        pipeline2 = FeaturePipeline(config=sample_config)
        pipeline2.fit(sample_dataframe, sample_target)
        result2 = pipeline2.transform(sample_dataframe)

        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_single_row(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test transforming a single row as dict."""
        row = sample_dataframe.iloc[0].to_dict()

        result = fitted_pipeline.transform_single(row)

        assert isinstance(result, dict)
        assert len(result) == len(fitted_pipeline.get_feature_names())

    def test_transform_single_matches_batch(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test that single-row transform matches batch transform."""
        row = sample_dataframe.iloc[0].to_dict()

        single_result = fitted_pipeline.transform_single(row)
        batch_result = fitted_pipeline.transform(sample_dataframe.iloc[[0]])

        for col in fitted_pipeline.get_feature_names():
            assert single_result[col] == pytest.approx(
                batch_result[col].iloc[0], rel=1e-5
            )

    def test_derived_features(
        self,
        sample_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
        sample_target: pd.Series,
    ) -> None:
        """Test that derived features are computed correctly."""
        pipeline = FeaturePipeline(config=sample_config)
        result = pipeline.fit_transform(sample_dataframe, sample_target)

        # Check derived feature exists
        assert "loan_to_income_ratio" in result.columns

        # Derived features use transformed values from result DataFrame
        # (result takes precedence over original in the namespace merge)
        # Formula: "loan_amnt / (annual_inc + 1)" uses log1p-transformed values
        loan_amnt_transformed = result["loan_amnt"]
        annual_inc_transformed = result["annual_inc"]
        expected = loan_amnt_transformed / (annual_inc_transformed + 1)

        np.testing.assert_array_almost_equal(
            result["loan_to_income_ratio"].values,
            expected.values,
            decimal=5,
        )

    def test_missing_column_handling(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test handling of missing columns during transform."""
        # Remove a column
        incomplete_df = sample_dataframe.drop(columns=["loan_amnt"])

        result = fitted_pipeline.transform(incomplete_df)

        # Should still produce output with default values
        assert len(result.columns) == len(fitted_pipeline.get_feature_names())
        assert "loan_amnt" in result.columns

    def test_transform_before_fit_raises(
        self,
        sample_config: dict[str, Any],
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test that transforming before fitting raises error."""
        pipeline = FeaturePipeline(config=sample_config)

        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.transform(sample_dataframe)


class TestPipelineSerialization:
    """Tests for pipeline save/load functionality."""

    def test_save_load_roundtrip(
        self,
        fitted_pipeline: FeaturePipeline,
        sample_dataframe: pd.DataFrame,
        temp_dir: Path,
    ) -> None:
        """Test that saved pipeline can be loaded and produces same results."""
        # Save
        save_path = temp_dir / "pipeline.json"
        fitted_pipeline.save(save_path)

        # Load
        loaded_pipeline = FeaturePipeline.load(save_path)

        # Compare transforms
        original_result = fitted_pipeline.transform(sample_dataframe)
        loaded_result = loaded_pipeline.transform(sample_dataframe)

        pd.testing.assert_frame_equal(original_result, loaded_result)

    def test_save_creates_directories(
        self,
        fitted_pipeline: FeaturePipeline,
        temp_dir: Path,
    ) -> None:
        """Test that save creates parent directories."""
        save_path = temp_dir / "nested" / "path" / "pipeline.json"

        fitted_pipeline.save(save_path)

        assert save_path.exists()

    def test_load_nonexistent_raises(self, temp_dir: Path) -> None:
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FeaturePipeline.load(temp_dir / "nonexistent.json")

    def test_feature_names_preserved(
        self,
        fitted_pipeline: FeaturePipeline,
        temp_dir: Path,
    ) -> None:
        """Test that feature names are preserved after save/load."""
        save_path = temp_dir / "pipeline.json"
        fitted_pipeline.save(save_path)

        loaded_pipeline = FeaturePipeline.load(save_path)

        assert loaded_pipeline.get_feature_names() == fitted_pipeline.get_feature_names()


class TestFeatureImportanceMapping:
    """Tests for feature importance mapping."""

    def test_importance_mapping(
        self,
        fitted_pipeline: FeaturePipeline,
    ) -> None:
        """Test mapping importances to feature names."""
        n_features = len(fitted_pipeline.get_feature_names())
        importances = np.random.rand(n_features)

        mapping = fitted_pipeline.get_feature_importance_map(importances)

        assert len(mapping) == n_features
        for name in fitted_pipeline.get_feature_names():
            assert name in mapping

    def test_importance_wrong_length_raises(
        self,
        fitted_pipeline: FeaturePipeline,
    ) -> None:
        """Test that wrong-length importance array raises error."""
        importances = np.array([1, 2, 3])  # Wrong length

        with pytest.raises(ValueError, match="doesn't match"):
            fitted_pipeline.get_feature_importance_map(importances)
