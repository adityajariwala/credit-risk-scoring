"""
Feature transformations for credit risk scoring.

This module provides reusable transformations that maintain parity between
offline training and online inference. Each transform is a pure function
that can be applied consistently across both environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BaseTransform(ABC):
    """Abstract base class for feature transformations."""

    @abstractmethod
    def fit(self, data: pd.Series) -> BaseTransform:
        """Fit the transform on training data."""
        pass

    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series | pd.DataFrame:
        """Apply the transform to data."""
        pass

    def fit_transform(self, data: pd.Series) -> pd.Series | pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize transform state for persistence."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> BaseTransform:
        """Deserialize transform from persisted state."""
        pass


@dataclass
class Log1pTransform(BaseTransform):
    """Log(1+x) transform for handling skewed distributions."""

    fill_value: float = 0.0
    _fitted: bool = False

    def fit(self, data: pd.Series) -> Log1pTransform:
        self._fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        filled = data.fillna(self.fill_value)
        # Clip negative values to 0 before log
        return pd.Series(np.log1p(np.maximum(filled, 0)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "log1p", "fill_value": self.fill_value}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Log1pTransform:
        transform = cls(fill_value=config.get("fill_value", 0.0))
        transform._fitted = True
        return transform


@dataclass
class ClipTransform(BaseTransform):
    """Clip values to a specified range."""

    clip_min: float
    clip_max: float
    fill_value: float = 0.0
    _fitted: bool = False

    def fit(self, data: pd.Series) -> ClipTransform:
        self._fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        filled = data.fillna(self.fill_value)
        return pd.Series(np.clip(filled, self.clip_min, self.clip_max))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "clip",
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "fill_value": self.fill_value,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ClipTransform:
        transform = cls(
            clip_min=config["clip_min"],
            clip_max=config["clip_max"],
            fill_value=config.get("fill_value", 0.0),
        )
        transform._fitted = True
        return transform


@dataclass
class IdentityTransform(BaseTransform):
    """No-op transform, just handles missing values."""

    fill_value: float = 0.0
    _fitted: bool = False

    def fit(self, data: pd.Series) -> IdentityTransform:
        self._fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        return data.fillna(self.fill_value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "none", "fill_value": self.fill_value}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> IdentityTransform:
        transform = cls(fill_value=config.get("fill_value", 0.0))
        transform._fitted = True
        return transform


@dataclass
class StandardScalerTransform(BaseTransform):
    """Standardize features by removing mean and scaling to unit variance."""

    fill_value: float = 0.0
    mean_: float | None = None
    std_: float | None = None

    def fit(self, data: pd.Series) -> StandardScalerTransform:
        filled = data.fillna(self.fill_value)
        self.mean_ = float(filled.mean())
        self.std_ = float(filled.std())
        # Prevent division by zero
        if self.std_ == 0:
            self.std_ = 1.0
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Transform must be fitted before transform")
        filled = data.fillna(self.fill_value)
        return (filled - self.mean_) / self.std_

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "standard_scaler",
            "fill_value": self.fill_value,
            "mean": self.mean_,
            "std": self.std_,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> StandardScalerTransform:
        transform = cls(fill_value=config.get("fill_value", 0.0))
        transform.mean_ = config.get("mean")
        transform.std_ = config.get("std")
        return transform


@dataclass
class LabelEncoderTransform(BaseTransform):
    """Encode categorical values as integers."""

    categories: list[str]
    unknown_value: int = -1
    _mapping: dict[str, int] | None = None

    def fit(self, data: pd.Series) -> LabelEncoderTransform:
        self._mapping = {cat: idx for idx, cat in enumerate(self.categories)}
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        if self._mapping is None:
            raise ValueError("Transform must be fitted before transform")
        else:
            return data.map(lambda x: (self._mapping or {}).get(x, self.unknown_value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "label",
            "categories": self.categories,
            "unknown_value": self.unknown_value,
            "mapping": self._mapping,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LabelEncoderTransform:
        transform = cls(
            categories=config["categories"],
            unknown_value=config.get("unknown_value", -1),
        )
        transform._mapping = config.get("mapping")
        return transform


@dataclass
class OrdinalEncoderTransform(BaseTransform):
    """Encode ordinal categorical values preserving order."""

    categories: list[str]
    unknown_value: int = -1
    _mapping: dict[str, int] | None = None

    def fit(self, data: pd.Series) -> OrdinalEncoderTransform:
        # Ordinal encoding preserves the order of categories
        self._mapping = {cat: idx for idx, cat in enumerate(self.categories)}
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        if self._mapping is None:
            raise ValueError("Transform must be fitted before transform")
        return data.map(lambda x: (self._mapping or {}).get(x, self.unknown_value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ordinal",
            "categories": self.categories,
            "unknown_value": self.unknown_value,
            "mapping": self._mapping,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> OrdinalEncoderTransform:
        transform = cls(
            categories=config["categories"],
            unknown_value=config.get("unknown_value", -1),
        )
        transform._mapping = config.get("mapping")
        return transform


@dataclass
class OneHotEncoderTransform(BaseTransform):
    """One-hot encode categorical values."""

    categories: list[str]
    feature_name: str = ""
    _fitted: bool = False

    def fit(self, data: pd.Series) -> OneHotEncoderTransform:
        self._fitted = True
        return self

    def transform(self, data: pd.Series) -> pd.DataFrame:
        """Returns a DataFrame with one column per category."""
        if not self._fitted:
            raise ValueError("Transform must be fitted before transform")

        result = pd.DataFrame(index=data.index)
        for cat in self.categories:
            col_name = f"{self.feature_name}_{cat}" if self.feature_name else cat
            result[col_name] = (data == cat).astype(int)
        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names after transformation."""
        return [
            f"{self.feature_name}_{cat}" if self.feature_name else cat for cat in self.categories
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "onehot",
            "categories": self.categories,
            "feature_name": self.feature_name,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> OneHotEncoderTransform:
        transform = cls(
            categories=config["categories"],
            feature_name=config.get("feature_name", ""),
        )
        transform._fitted = True
        return transform


@dataclass
class TargetEncoderTransform(BaseTransform):
    """Target encoding for high-cardinality categorical features."""

    min_samples: int = 100
    smoothing: float = 1.0
    fill_value: float | None = None
    _encoding_map: dict[str, float] | None = None
    _global_mean: float | None = None

    def fit(self, data: pd.Series, target: pd.Series | None = None) -> TargetEncoderTransform:
        if target is None:
            raise ValueError("Target encoding requires target values for fitting")

        self._global_mean = float(target.mean())

        # Calculate category means with smoothing
        df = pd.DataFrame({"cat": data, "target": target})
        stats = df.groupby("cat")["target"].agg(["mean", "count"])

        self._encoding_map = {}
        for cat, row in stats.iterrows():
            if row["count"] >= self.min_samples:
                # Apply smoothing: weighted average of category mean and global mean
                weight = row["count"] / (row["count"] + self.smoothing)
                smoothed_mean = weight * row["mean"] + (1 - weight) * self._global_mean
                self._encoding_map[str(cat)] = smoothed_mean
            else:
                # Use global mean for rare categories
                self._encoding_map[str(cat)] = self._global_mean

        self.fill_value = self._global_mean
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        if self._encoding_map is None or self._global_mean is None:
            raise ValueError("Transform must be fitted before transform")
        return data.map(lambda x: (self._encoding_map or {}).get(x, self._global_mean))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "target",
            "min_samples": self.min_samples,
            "smoothing": self.smoothing,
            "encoding_map": self._encoding_map,
            "global_mean": self._global_mean,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TargetEncoderTransform:
        transform = cls(
            min_samples=config.get("min_samples", 100),
            smoothing=config.get("smoothing", 1.0),
        )
        transform._encoding_map = config.get("encoding_map")
        transform._global_mean = config.get("global_mean")
        transform.fill_value = transform._global_mean
        return transform


def create_transform(config: dict[str, Any]) -> BaseTransform:
    """Factory function to create transforms from config."""
    transform_type = config.get("transform", config.get("type", "none"))

    if transform_type == "log1p":
        return Log1pTransform(fill_value=config.get("fill_value", 0.0))
    elif transform_type == "clip":
        return ClipTransform(
            clip_min=config.get("clip_min", 0),
            clip_max=config.get("clip_max", 100),
            fill_value=config.get("fill_value", 0.0),
        )
    elif transform_type == "standard_scaler":
        return StandardScalerTransform(fill_value=config.get("fill_value", 0.0))
    elif transform_type == "label":
        return LabelEncoderTransform(
            categories=config.get("categories", []),
            unknown_value=config.get("unknown_value", -1),
        )
    elif transform_type == "ordinal":
        return OrdinalEncoderTransform(
            categories=config.get("categories", []),
            unknown_value=config.get("unknown_value", -1),
        )
    elif transform_type == "onehot":
        return OneHotEncoderTransform(
            categories=config.get("categories", []),
            feature_name=config.get("name", ""),
        )
    elif transform_type == "target":
        return TargetEncoderTransform(
            min_samples=config.get("min_samples", 100),
            smoothing=config.get("smoothing", 1.0),
        )
    else:
        return IdentityTransform(fill_value=config.get("fill_value", 0.0))


def deserialize_transform(config: dict[str, Any]) -> BaseTransform:
    """Deserialize a transform from its persisted state."""
    transform_type = config.get("type", "none")

    if transform_type == "log1p":
        return Log1pTransform.from_dict(config)
    elif transform_type == "clip":
        return ClipTransform.from_dict(config)
    elif transform_type == "standard_scaler":
        return StandardScalerTransform.from_dict(config)
    elif transform_type == "label":
        return LabelEncoderTransform.from_dict(config)
    elif transform_type == "ordinal":
        return OrdinalEncoderTransform.from_dict(config)
    elif transform_type == "onehot":
        return OneHotEncoderTransform.from_dict(config)
    elif transform_type == "target":
        return TargetEncoderTransform.from_dict(config)
    else:
        return IdentityTransform.from_dict(config)
