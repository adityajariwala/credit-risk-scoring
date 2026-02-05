"""
Unified feature pipeline — same code path for training and serving.

The key idea: one FeaturePipeline instance gets fit during training,
serialized alongside the model, and reloaded at inference time. This
eliminates the train/serve skew problem.
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
    """Orchestrates numeric, categorical, and derived feature transforms."""

    config: dict[str, Any]
    numeric_transforms: dict[str, BaseTransform] = field(default_factory=dict)
    categorical_transforms: dict[str, BaseTransform] = field(default_factory=dict)
    derived_features: list[dict[str, str]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    _fitted: bool = False

    def __post_init__(self) -> None:
        self.derived_features = self.config.get("features", {}).get("derived", [])

    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> FeaturePipeline:
        """Fit all transforms on training data. Pass target if using target encoding."""
        # numeric
        for feat_config in self.config.get("features", {}).get("numeric", []):
            name = feat_config["name"]
            if name in df.columns:
                transform = create_transform(feat_config)
                transform.fit(df[name])
                self.numeric_transforms[name] = transform

        # categorical
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

        self._build_feature_names()
        self._fitted = True
        return self

    def _build_feature_names(self) -> None:
        """Assemble final column ordering (numeric -> categorical -> derived)."""
        self.feature_names = []

        for name in self.numeric_transforms:
            self.feature_names.append(name)

        for name, transform in self.categorical_transforms.items():
            if isinstance(transform, OneHotEncoderTransform):
                self.feature_names.extend(transform.get_feature_names())
            else:
                self.feature_names.append(name)

        for derived in self.derived_features:
            self.feature_names.append(derived["name"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transforms, returning columns in deterministic order."""
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        result_parts: list[pd.DataFrame | pd.Series] = []

        for name, transform in self.numeric_transforms.items():
            if name in df.columns:
                transformed = transform.transform(df[name])
                transformed.name = name
                result_parts.append(transformed)
            else:
                result_parts.append(pd.Series(0, index=df.index, name=name))

        for name, transform in self.categorical_transforms.items():
            if name in df.columns:
                transformed = transform.transform(df[name])
                if isinstance(transformed, pd.DataFrame):
                    result_parts.append(transformed)
                else:
                    transformed.name = name
                    result_parts.append(transformed)
            else:
                if isinstance(transform, OneHotEncoderTransform):
                    for col_name in transform.get_feature_names():
                        result_parts.append(pd.Series(0, index=df.index, name=col_name))
                else:
                    result_parts.append(pd.Series(-1, index=df.index, name=name))

        result = pd.concat(result_parts, axis=1) if result_parts else pd.DataFrame(index=df.index)
        result = self._add_derived_features(result, df)
        return result[self.feature_names]

    def _add_derived_features(self, result: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Evaluate config-driven formulas. Falls back to 0 if something explodes."""
        for derived in self.derived_features:
            name = derived["name"]
            formula = derived["formula"]

            try:
                # merge both namespaces so formulas can reference either raw or transformed cols
                namespace = {**original.to_dict("series"), **result.to_dict("series"), "np": np}
                result[name] = pd.eval(
                    formula, local_dict={str(k): v for k, v in namespace.items()}
                )
                result[name] = result[name].replace([np.inf, -np.inf], 0).fillna(0)
            except Exception as e:
                result[name] = 0
                # TODO: swap this for structlog once we add it to the pipeline module
                print(f"Warning: Failed to compute derived feature {name}: {e}")

        return result

    def fit_transform(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, target).transform(df)

    def transform_single(self, row: dict[str, Any]) -> dict[str, float]:
        """Single-row transform for real-time serving. Not the fastest thing
        ever (still builds a 1-row DataFrame) but keeps parity simple."""
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        df = pd.DataFrame([row])
        transformed = self.transform(df)

        return {str(k): v for k, v in transformed.iloc[0].to_dict().items()}

    def save(self, path: str | Path) -> None:
        """Persist all transform states to JSON."""
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
        """Reconstruct a fitted pipeline from a JSON file saved by .save()."""
        path = Path(path)

        with open(path) as f:
            state = json.load(f)

        pipeline = cls(config=config or {})

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
        return self.feature_names.copy()

    def get_feature_importance_map(self, importances: np.ndarray) -> dict[str, float]:
        """Zip feature importances with their names for readability."""
        if len(importances) != len(self.feature_names):
            raise ValueError(
                f"Importance array length ({len(importances)}) doesn't match "
                f"number of features ({len(self.feature_names)})"
            )

        return dict(zip(self.feature_names, importances, strict=True))
