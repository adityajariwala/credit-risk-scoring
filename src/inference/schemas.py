"""
Pydantic schemas for API request/response validation.

These schemas provide type safety, validation, and automatic OpenAPI documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskTier(str, Enum):
    """Risk tier classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Recommendation(str, Enum):
    """Lending recommendation based on risk tier."""

    APPROVE = "approve"
    REVIEW = "review"
    ENHANCED_REVIEW = "enhanced_review"
    DECLINE = "decline"


class LoanApplication(BaseModel):
    """
    Input schema for credit risk prediction.

    All fields correspond to features used in the model.
    """

    loan_amnt: float = Field(..., gt=0, le=50000, description="Loan amount in USD")
    annual_inc: float = Field(..., gt=0, le=1000000, description="Annual income in USD")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio")
    open_acc: int = Field(..., ge=0, le=50, description="Number of open credit accounts")
    revol_bal: float = Field(..., ge=0, description="Revolving balance")
    revol_util: float = Field(..., ge=0, le=150, description="Revolving utilization rate")
    total_acc: int = Field(..., ge=0, le=100, description="Total number of credit accounts")
    int_rate: float = Field(..., ge=0, le=35, description="Interest rate")
    installment: float = Field(..., gt=0, description="Monthly installment amount")
    term: str = Field(..., description="Loan term (36 months or 60 months)")
    grade: str = Field(..., description="Loan grade (A-G)")
    home_ownership: str = Field(..., description="Home ownership status")
    verification_status: str = Field(..., description="Income verification status")
    purpose: str = Field(..., description="Loan purpose")

    @field_validator("term")
    @classmethod
    def validate_term(cls, v: str) -> str:
        valid_terms = ["36 months", "60 months"]
        if v not in valid_terms:
            raise ValueError(f"term must be one of {valid_terms}")
        return v

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        valid_grades = ["A", "B", "C", "D", "E", "F", "G"]
        if v not in valid_grades:
            raise ValueError(f"grade must be one of {valid_grades}")
        return v

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        valid_values = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        if v not in valid_values:
            raise ValueError(f"home_ownership must be one of {valid_values}")
        return v

    @field_validator("verification_status")
    @classmethod
    def validate_verification_status(cls, v: str) -> str:
        valid_values = ["Verified", "Source Verified", "Not Verified"]
        if v not in valid_values:
            raise ValueError(f"verification_status must be one of {valid_values}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class FeatureContribution(BaseModel):
    """Single feature's contribution to the prediction."""

    feature: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Feature value")
    contribution: float = Field(..., description="SHAP contribution to prediction")


class PredictionExplanation(BaseModel):
    """SHAP-based explanation for the prediction."""

    base_value: float = Field(..., description="Expected model output (base prediction)")
    top_contributors: list[FeatureContribution] = Field(
        ..., description="Top features contributing to the prediction"
    )


class RiskPrediction(BaseModel):
    """
    Output schema for credit risk prediction.

    Includes risk score, classification, and optional explanation.
    """

    risk_score: float = Field(
        ..., ge=0, le=1, description="Probability of default (0-1)"
    )
    risk_tier: RiskTier = Field(..., description="Risk tier classification")
    recommendation: Recommendation = Field(
        ..., description="Lending recommendation based on risk tier"
    )
    model_version: str = Field(..., description="Version of the model used")
    explanation: PredictionExplanation | None = Field(
        None, description="SHAP-based explanation (if enabled)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "risk_score": 0.35,
                "risk_tier": "medium",
                "recommendation": "review",
                "model_version": "v1.0.0",
                "explanation": {
                    "base_value": 0.15,
                    "top_contributors": [
                        {"feature": "dti", "value": 35.5, "contribution": 0.12},
                        {"feature": "grade", "value": "D", "contribution": 0.08},
                    ],
                },
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    applications: list[LoanApplication] = Field(
        ..., min_length=1, max_length=100, description="List of loan applications"
    )
    include_explanations: bool = Field(
        False, description="Whether to include SHAP explanations"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[RiskPrediction] = Field(..., description="List of predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")


class ModelInfo(BaseModel):
    """Model metadata response."""

    model_name: str
    model_version: str
    model_type: str
    n_features: int
    feature_names: list[str]
    trained_at: str
    metrics: dict[str, Any]
