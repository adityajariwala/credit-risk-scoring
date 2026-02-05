"""
Unit tests for feature transformations.

Tests individual transform classes for correctness and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transforms import (
    ClipTransform,
    IdentityTransform,
    LabelEncoderTransform,
    Log1pTransform,
    OneHotEncoderTransform,
    OrdinalEncoderTransform,
    StandardScalerTransform,
    TargetEncoderTransform,
    create_transform,
    deserialize_transform,
)


class TestLog1pTransform:
    """Tests for Log1pTransform."""

    def test_transform_positive_values(self) -> None:
        """Test log1p transform on positive values."""
        transform = Log1pTransform()
        data = pd.Series([0, 1, 10, 100, 1000])

        result = transform.fit(data).transform(data)

        expected = np.log1p(data)
        pd.testing.assert_series_equal(result, expected)

    def test_transform_with_nan(self) -> None:
        """Test handling of NaN values."""
        transform = Log1pTransform(fill_value=0)
        data = pd.Series([1, np.nan, 10, np.nan])

        result = transform.fit(data).transform(data)

        assert not result.isna().any()
        assert result.iloc[1] == np.log1p(0)

    def test_transform_negative_values(self) -> None:
        """Test that negative values are clipped to 0."""
        transform = Log1pTransform()
        data = pd.Series([-10, -1, 0, 1])

        result = transform.fit(data).transform(data)

        # Negative values should be clipped to 0 before log
        assert result.iloc[0] == np.log1p(0)
        assert result.iloc[1] == np.log1p(0)

    def test_serialization(self) -> None:
        """Test transform can be serialized and deserialized."""
        transform = Log1pTransform(fill_value=5)
        transform.fit(pd.Series([1, 2, 3]))

        serialized = transform.to_dict()
        restored = Log1pTransform.from_dict(serialized)

        assert restored.fill_value == transform.fill_value


class TestClipTransform:
    """Tests for ClipTransform."""

    def test_clip_values(self) -> None:
        """Test clipping values within range."""
        transform = ClipTransform(clip_min=10, clip_max=90)
        data = pd.Series([0, 50, 100])

        result = transform.fit(data).transform(data)

        assert result.iloc[0] == 10
        assert result.iloc[1] == 50
        assert result.iloc[2] == 90

    def test_clip_with_nan(self) -> None:
        """Test NaN handling in clip transform."""
        transform = ClipTransform(clip_min=0, clip_max=100, fill_value=50)
        data = pd.Series([25, np.nan, 75])

        result = transform.fit(data).transform(data)

        assert result.iloc[1] == 50

    def test_serialization(self) -> None:
        """Test clip transform serialization."""
        transform = ClipTransform(clip_min=0, clip_max=100, fill_value=10)
        transform.fit(pd.Series([1, 2, 3]))

        serialized = transform.to_dict()
        restored = ClipTransform.from_dict(serialized)

        assert restored.clip_min == transform.clip_min
        assert restored.clip_max == transform.clip_max


class TestIdentityTransform:
    """Tests for IdentityTransform."""

    def test_identity_returns_same_values(self) -> None:
        """Test that identity transform preserves values."""
        transform = IdentityTransform()
        data = pd.Series([1.5, 2.5, 3.5])

        result = transform.fit(data).transform(data)

        pd.testing.assert_series_equal(result, data)

    def test_identity_fills_nan(self) -> None:
        """Test NaN filling."""
        transform = IdentityTransform(fill_value=-1)
        data = pd.Series([1, np.nan, 3])

        result = transform.fit(data).transform(data)

        assert result.iloc[1] == -1


class TestStandardScalerTransform:
    """Tests for StandardScalerTransform."""

    def test_standardization(self) -> None:
        """Test that output has mean ~0 and std ~1."""
        transform = StandardScalerTransform()
        data = pd.Series([10, 20, 30, 40, 50])

        result = transform.fit(data).transform(data)

        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1) < 1e-10

    def test_constant_series(self) -> None:
        """Test handling of constant series (zero std)."""
        transform = StandardScalerTransform()
        data = pd.Series([5, 5, 5, 5])

        result = transform.fit(data).transform(data)

        # Should not raise division by zero
        assert not result.isna().any()

    def test_serialization(self) -> None:
        """Test scaler serialization preserves parameters."""
        transform = StandardScalerTransform()
        data = pd.Series([1, 2, 3, 4, 5])
        transform.fit(data)

        serialized = transform.to_dict()
        restored = StandardScalerTransform.from_dict(serialized)

        assert restored.mean_ == transform.mean_
        assert restored.std_ == transform.std_


class TestLabelEncoderTransform:
    """Tests for LabelEncoderTransform."""

    def test_encoding(self) -> None:
        """Test label encoding of categories."""
        transform = LabelEncoderTransform(categories=["A", "B", "C"])
        data = pd.Series(["A", "B", "C", "A"])

        result = transform.fit(data).transform(data)

        assert list(result) == [0, 1, 2, 0]

    def test_unknown_category(self) -> None:
        """Test handling of unknown categories."""
        transform = LabelEncoderTransform(categories=["A", "B"], unknown_value=-1)
        data = pd.Series(["A", "X", "B"])

        result = transform.fit(data).transform(data)

        assert result.iloc[1] == -1


class TestOrdinalEncoderTransform:
    """Tests for OrdinalEncoderTransform."""

    def test_ordinal_encoding_preserves_order(self) -> None:
        """Test that ordinal encoding preserves category order."""
        categories = ["low", "medium", "high"]
        transform = OrdinalEncoderTransform(categories=categories)
        data = pd.Series(["high", "low", "medium"])

        result = transform.fit(data).transform(data)

        assert result.iloc[0] == 2  # high
        assert result.iloc[1] == 0  # low
        assert result.iloc[2] == 1  # medium


class TestOneHotEncoderTransform:
    """Tests for OneHotEncoderTransform."""

    def test_one_hot_encoding(self) -> None:
        """Test one-hot encoding creates correct columns."""
        transform = OneHotEncoderTransform(categories=["A", "B", "C"], feature_name="cat")
        data = pd.Series(["A", "B", "A", "C"])

        result = transform.fit(data).transform(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["cat_A", "cat_B", "cat_C"]
        assert list(result["cat_A"]) == [1, 0, 1, 0]
        assert list(result["cat_B"]) == [0, 1, 0, 0]

    def test_get_feature_names(self) -> None:
        """Test feature name generation."""
        transform = OneHotEncoderTransform(categories=["X", "Y"], feature_name="test")
        transform.fit(pd.Series(["X", "Y"]))

        names = transform.get_feature_names()

        assert names == ["test_X", "test_Y"]


class TestTargetEncoderTransform:
    """Tests for TargetEncoderTransform."""

    def test_target_encoding(self) -> None:
        """Test target encoding with sufficient samples."""
        transform = TargetEncoderTransform(min_samples=2)
        data = pd.Series(["A", "A", "B", "B", "A"])
        target = pd.Series([1, 1, 0, 0, 1])  # A has 100% rate, B has 0%

        result = transform.fit(data, target).transform(data)

        # A values should be higher than B values
        assert result.iloc[0] > result.iloc[2]

    def test_rare_category_uses_global_mean(self) -> None:
        """Test that rare categories fall back to global mean."""
        transform = TargetEncoderTransform(min_samples=10)
        data = pd.Series(["common"] * 20 + ["rare"])
        target = pd.Series([0.5] * 21)

        transform.fit(data, target)

        # Rare category should use global mean
        assert transform._encoding_map["rare"] == pytest.approx(transform._global_mean, rel=0.1)


class TestTransformFactory:
    """Tests for transform factory functions."""

    def test_create_transform(self) -> None:
        """Test creating transforms from config."""
        configs = [
            {"transform": "log1p"},
            {"transform": "clip", "clip_min": 0, "clip_max": 100},
            {"transform": "none"},
            {"type": "label", "categories": ["A", "B"]},
        ]

        for config in configs:
            transform = create_transform(config)
            assert transform is not None

    def test_deserialize_transform(self) -> None:
        """Test deserializing transforms from saved state."""
        original = ClipTransform(clip_min=0, clip_max=50, fill_value=25)
        original.fit(pd.Series([1, 2, 3]))

        serialized = original.to_dict()
        restored = deserialize_transform(serialized)

        assert isinstance(restored, ClipTransform)
        assert restored.clip_min == original.clip_min
        assert restored.clip_max == original.clip_max
