"""
Feature engineering pipeline for credit risk scoring.

This module provides a unified pipeline that ensures feature parity between
offline training and online inference. The same transformations are applied
consistently, preventing training-serving skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.transforms import (
    BaseTransform,
    OneHotEncoderTransform,
    TargetEncoderTransform,
    create_transform,
    deserialize_transform,
)


@dataclass
class FeaturePipeline:
    """
    Unified feature engineering pipeline.

    Ensures identical transformations during training and inference,
    preventing feature/training/serving skew.
    """

    config: dict[str, Any]
    numeric_transforms: dict[str, BaseTransform] = field(default_factory=dict)
    categorical_transforms: dict[str, BaseTransform] = field(default_factory=dict)
    derived_features: list[dict[str, str]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    _fitted: bool = False

    def __post_init__(self) -> None:
        """Initialize transforms from config."""
        self.derived_features = self.config.get("features", {}).get("derived", [])

    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> FeaturePipeline:
        """
        Fit all transformations on training data.

        Args:
            df: Training DataFrame
            target: Target variable (required for target encoding)

        Returns:
            Self for method chaining
        """
        # Fit numeric transforms
        for feat_config in self.config.get("features", {}).get("numeric", []):
            name = feat_config["name"]
            if name in df.columns:
                transform = create_transform(feat_config)
                transform.fit(df[name])
                self.numeric_transforms[name] = transform

        # Fit categorical transforms
        for feat_config in self.config.get("features", {}).get("categorical", []):
            name = feat_config["name"]
            encoding = feat_config.get("encoding", "label")

            if name in df.columns:
                transform = create_transform({"type": encoding, **feat_config})

                if isinstance(transform, TargetEncoderTransform):
                    if target is None:
                        raise ValueError(f"Target encoding for {name} requires target values")
                    transform.fit(df[name], target)
                else:
                    transform.fit(df[name])

                self.categorical_transforms[name] = transform

        # Build final feature name list
        self._build_feature_names()
        self._fitted = True
        return self

    def _build_feature_names(self) -> None:
        """Build the list of output feature names after transformation."""
        self.feature_names = []

        # Numeric features
        for name in self.numeric_transforms:
            self.feature_names.append(name)

        # Categorical features (handle one-hot expansion)
        for name, transform in self.categorical_transforms.items():
            if isinstance(transform, OneHotEncoderTransform):
                self.feature_names.extend(transform.get_feature_names())
            else:
                self.feature_names.append(name)

        # Derived features
        for derived in self.derived_features:
            self.feature_names.append(derived["name"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame with consistent feature ordering
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        result_parts: list[pd.DataFrame | pd.Series] = []

        # Transform numeric features
        for name, transform in self.numeric_transforms.items():
            if name in df.columns:
                transformed = transform.transform(df[name])
                transformed.name = name
                result_parts.append(transformed)
            else:
                # Fill missing column with default
                result_parts.append(pd.Series(0, index=df.index, name=name))

        # Transform categorical features
        for name, transform in self.categorical_transforms.items():
            if name in df.columns:
                transformed = transform.transform(df[name])
                if isinstance(transformed, pd.DataFrame):
                    result_parts.append(transformed)
                else:
                    transformed.name = name
                    result_parts.append(transformed)
            else:
                # Fill missing categorical with default
                if isinstance(transform, OneHotEncoderTransform):
                    for col_name in transform.get_feature_names():
                        result_parts.append(pd.Series(0, index=df.index, name=col_name))
                else:
                    result_parts.append(pd.Series(-1, index=df.index, name=name))

        # Concatenate all transformed features
        result = pd.concat(result_parts, axis=1) if result_parts else pd.DataFrame(index=df.index)

        # Add derived features
        result = self._add_derived_features(result, df)

        # Ensure consistent column ordering
        return result[self.feature_names]

    def _add_derived_features(self, result: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features based on formulas.

        Uses the original DataFrame for base feature values and
        adds computed features to the result.
        """
        for derived in self.derived_features:
            name = derived["name"]
            formula = derived["formula"]

            try:
                # Create a combined namespace for eval
                namespace = {**original.to_dict("series"), **result.to_dict("series"), "np": np}

                # Evaluate the formula safely
                result[name] = pd.eval(
                    formula, local_dict={str(k): v for k, v in namespace.items()}
                )

                # Handle any infinities or NaN from division
                result[name] = result[name].replace([np.inf, -np.inf], 0).fillna(0)

            except Exception as e:
                # If formula evaluation fails, fill with zeros
                result[name] = 0
                # Log warning in production
                print(f"Warning: Failed to compute derived feature {name}: {e}")

        return result

    def fit_transform(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target).transform(df)

    def transform_single(self, row: dict[str, Any]) -> dict[str, float]:
        """
        Transform a single observation for real-time inference.

        This method is optimized for low-latency single-row transformation.

        Args:
            row: Dictionary with feature values

        Returns:
            Dictionary with transformed feature values
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        # Convert to single-row DataFrame for consistent processing
        df = pd.DataFrame([row])
        transformed = self.transform(df)

        return {str(k): v for k, v in transformed.iloc[0].to_dict().items()}

    def save(self, path: str | Path) -> None:
        """
        Save fitted pipeline to disk.

        Serializes all transform states for later loading.
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "numeric_transforms": {
                name: transform.to_dict() for name, transform in self.numeric_transforms.items()
            },
            "categorical_transforms": {
                name: transform.to_dict() for name, transform in self.categorical_transforms.items()
            },
            "derived_features": self.derived_features,
            "feature_names": self.feature_names,
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, config: dict[str, Any] | None = None) -> FeaturePipeline:
        """
        Load a fitted pipeline from disk.

        Args:
            path: Path to saved pipeline state
            config: Optional config (for reference, not required for transform)

        Returns:
            Loaded and ready-to-use pipeline
        """
        path = Path(path)

        with open(path) as f:
            state = json.load(f)

        pipeline = cls(config=config or {})

        # Deserialize transforms
        pipeline.numeric_transforms = {
            name: deserialize_transform(transform_dict)
            for name, transform_dict in state["numeric_transforms"].items()
        }

        pipeline.categorical_transforms = {
            name: deserialize_transform(transform_dict)
            for name, transform_dict in state["categorical_transforms"].items()
        }

        pipeline.derived_features = state["derived_features"]
        pipeline.feature_names = state["feature_names"]
        pipeline._fitted = True

        return pipeline

    def get_feature_names(self) -> list[str]:
        """Get the list of output feature names."""
        return self.feature_names.copy()

    def get_feature_importance_map(self, importances: np.ndarray) -> dict[str, float]:
        """
        Map feature importances to feature names.

        Useful for model interpretation.
        """
        if len(importances) != len(self.feature_names):
            raise ValueError(
                f"Importance array length ({len(importances)}) doesn't match "
                f"number of features ({len(self.feature_names)})"
            )

        return dict(zip(self.feature_names, importances, strict=True))
