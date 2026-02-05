"""
Training pipeline for credit risk scoring model.

This module handles end-to-end model training including:
- Data loading and preprocessing
- Feature engineering via the unified pipeline
- Model training with hyperparameters from config
- Evaluation and metrics computation
- Model and artifact serialization
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluation.metrics import compute_threshold_metrics, evaluate_model
from src.features.pipeline import FeaturePipeline


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return {str(k): v for k, v in yaml.safe_load(f).items()}


def load_data(data_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare training data.

    Args:
        data_path: Path to the CSV data file
        config: Configuration dictionary

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = pd.read_csv(data_path)

    target_col = config["training"]["target_column"]
    positive_class = config["training"]["positive_class"]

    # Convert target to binary
    y = (df[target_col] == positive_class).astype(int)

    # Remove target and ID columns from features
    drop_cols = [target_col, "id", "member_id"] if "id" in df.columns else [target_col]
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols)

    return X, y


def create_sample_data(output_path: str | Path, n_samples: int = 10000) -> None:
    """
    Create synthetic sample data for demonstration.

    This generates realistic-looking credit data for testing the pipeline.
    In production, this would be replaced with actual data loading.
    """
    np.random.seed(42)

    data = {
        "loan_amnt": np.random.lognormal(9, 0.5, n_samples).clip(1000, 40000),
        "annual_inc": np.random.lognormal(11, 0.7, n_samples).clip(20000, 500000),
        "dti": np.random.beta(2, 5, n_samples) * 40,
        "open_acc": np.random.poisson(10, n_samples).clip(1, 30),
        "revol_bal": np.random.lognormal(8, 1.2, n_samples).clip(0, 100000),
        "revol_util": np.random.beta(2, 3, n_samples) * 100,
        "total_acc": np.random.poisson(20, n_samples).clip(1, 50),
        "int_rate": np.random.uniform(5, 25, n_samples),
        "installment": np.random.lognormal(6, 0.5, n_samples).clip(50, 1500),
        "term": np.random.choice(["36 months", "60 months"], n_samples, p=[0.75, 0.25]),
        "grade": np.random.choice(
            ["A", "B", "C", "D", "E", "F", "G"],
            n_samples,
            p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.07, 0.03],
        ),
        "home_ownership": np.random.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            n_samples,
            p=[0.35, 0.15, 0.45, 0.05],
        ),
        "verification_status": np.random.choice(
            ["Verified", "Source Verified", "Not Verified"],
            n_samples,
            p=[0.35, 0.35, 0.30],
        ),
        "purpose": np.random.choice(
            [
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
            n_samples,
            p=[0.45, 0.20, 0.10, 0.05, 0.05, 0.05, 0.03, 0.04, 0.03],
        ),
    }

    df = pd.DataFrame(data)

    # Generate target with realistic default rate (~15%)
    # Higher risk factors: high DTI, high interest rate, low income, grade D-G
    risk_score = (
        df["dti"] / 40 * 0.3
        + df["int_rate"] / 25 * 0.25
        + (1 - np.log1p(df["annual_inc"]) / 15) * 0.2
        + df["grade"].map({"A": 0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.5, "F": 0.7, "G": 0.9})
        * 0.25
    )
    risk_score = risk_score.fillna(0.5)
    default_prob = 1 / (1 + np.exp(-3 * (risk_score - 0.5)))
    df["loan_status"] = np.where(
        np.random.random(n_samples) < default_prob, "Charged Off", "Fully Paid"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample data with {n_samples} samples at {output_path}")
    print(f"Default rate: {(df['loan_status'] == 'Charged Off').mean():.2%}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
) -> lgb.LGBMClassifier:
    """
    Train LightGBM model with early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration

    Returns:
        Trained LightGBM classifier
    """
    model_params = config["model"]["params"].copy()

    # Handle class imbalance
    if config["training"].get("class_weight") == "balanced":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        model_params["scale_pos_weight"] = n_neg / n_pos

    # Extract early stopping rounds
    early_stopping_rounds = model_params.pop("early_stopping_rounds", 20)

    model = lgb.LGBMClassifier(**model_params)

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, config: dict[str, Any]) -> dict[str, list[float]]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        X: Features
        y: Labels
        config: Configuration

    Returns:
        Dictionary of metric lists across folds
    """
    n_folds = config["training"]["cv_folds"]
    random_state = config["training"]["random_state"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results: dict[str, list[float]] = {
        "auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train fold model
        model = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, config)

        # Evaluate
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        metrics = evaluate_model(y_val_fold, y_pred_proba)

        for metric, value in metrics.items():
            if metric in cv_results:
                cv_results[metric].append(value)

    return cv_results


def save_artifacts(
    model: lgb.LGBMClassifier,
    pipeline: FeaturePipeline,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Save all training artifacts.

    Args:
        model: Trained model
        pipeline: Fitted feature pipeline
        config: Configuration
        metrics: Evaluation metrics
        output_dir: Output directory

    Returns:
        Dictionary of artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = config["model"]["version"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create versioned subdirectory
    version_dir = output_dir / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save feature pipeline
    pipeline_path = version_dir / "pipeline.json"
    pipeline.save(pipeline_path)

    # Save model metadata
    metadata = {
        "model_name": config["model"]["name"],
        "model_version": model_version,
        "model_type": config["model"]["type"],
        "trained_at": timestamp,
        "feature_names": pipeline.get_feature_names(),
        "n_features": len(pipeline.get_feature_names()),
        "metrics": metrics,
        "config": config,
    }

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Create/update latest symlink
    latest_link = output_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(model_version, target_is_directory=True)

    print(f"\nArtifacts saved to {version_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Pipeline: {pipeline_path}")
    print(f"  - Metadata: {metadata_path}")

    return {
        "model": str(model_path),
        "pipeline": str(pipeline_path),
        "metadata": str(metadata_path),
        "version_dir": str(version_dir),
    }


def main() -> int:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train credit risk scoring model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for testing",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation (faster training)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Model: {config['model']['name']} v{config['model']['version']}")

    # Create sample data if requested or if data doesn't exist
    data_path = Path(args.data)
    if args.create_sample_data or not data_path.exists():
        print("\nCreating sample training data...")
        create_sample_data(data_path)

    # Load data
    print(f"\nLoading data from {data_path}")
    X, y = load_data(data_path, config)
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    test_size = config["training"]["test_size"]
    val_size = config["training"]["validation_size"]
    random_state = config["training"]["random_state"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp,
    )

    print("\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Create and fit feature pipeline
    print("\nFitting feature pipeline...")
    pipeline = FeaturePipeline(config=config)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    print(f"Transformed features: {X_train_transformed.shape[1]}")

    # Cross-validation
    if not args.skip_cv:
        print("\n" + "=" * 50)
        print("Cross-Validation")
        print("=" * 50)

        # Need to refit pipeline for CV on full train+val data
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])

        cv_pipeline = FeaturePipeline(config=config)
        X_cv_transformed = cv_pipeline.fit_transform(X_cv, y_cv)

        cv_results = cross_validate(X_cv_transformed, y_cv, config)

        print("\nCross-Validation Results:")
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

    # Train final model
    print("\n" + "=" * 50)
    print("Training Final Model")
    print("=" * 50)

    model = train_model(X_train_transformed, y_train, X_val_transformed, y_val, config)

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Test Set Evaluation")
    print("=" * 50)

    y_test_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred_proba)

    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Compute threshold-based metrics
    threshold_metrics = compute_threshold_metrics(
        y_test, y_test_pred_proba, config["inference"]["risk_tiers"]
    )
    print("\nThreshold Metrics:")
    for tier, tier_metrics in threshold_metrics.items():
        print(f"  {tier}: {tier_metrics}")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance_map = pipeline.get_feature_importance_map(model.feature_importances_)
    sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_importance[:10]:
        print(f"  {name}: {importance:.4f}")

    # Save artifacts
    all_metrics = {
        "test": test_metrics,
        "threshold": threshold_metrics,
        "cv": cv_results if not args.skip_cv else None,
    }

    save_artifacts(model, pipeline, config, all_metrics, args.output)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
