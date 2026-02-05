"""
Unit tests for API schemas.

Tests Pydantic model validation and serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.inference.schemas import (
    BatchPredictionRequest,
    FeatureContribution,
    LoanApplication,
    PredictionExplanation,
    Recommendation,
    RiskPrediction,
    RiskTier,
)


class TestLoanApplication:
    """Tests for LoanApplication schema."""

    def test_valid_application(self, sample_loan_application: dict) -> None:
        """Test creating a valid loan application."""
        app = LoanApplication(**sample_loan_application)

        assert app.loan_amnt == 15000.0
        assert app.grade == "B"

    def test_invalid_loan_amount(self, sample_loan_application: dict) -> None:
        """Test that negative loan amount is rejected."""
        sample_loan_application["loan_amnt"] = -1000

        with pytest.raises(ValidationError):
            LoanApplication(**sample_loan_application)

    def test_invalid_loan_amount_too_high(self, sample_loan_application: dict) -> None:
        """Test that loan amount over limit is rejected."""
        sample_loan_application["loan_amnt"] = 100000

        with pytest.raises(ValidationError):
            LoanApplication(**sample_loan_application)

    def test_invalid_grade(self, sample_loan_application: dict) -> None:
        """Test that invalid grade is rejected."""
        sample_loan_application["grade"] = "X"

        with pytest.raises(ValidationError, match="grade must be one of"):
            LoanApplication(**sample_loan_application)

    def test_invalid_term(self, sample_loan_application: dict) -> None:
        """Test that invalid term is rejected."""
        sample_loan_application["term"] = "24 months"

        with pytest.raises(ValidationError, match="term must be one of"):
            LoanApplication(**sample_loan_application)

    def test_invalid_home_ownership(self, sample_loan_application: dict) -> None:
        """Test that invalid home_ownership is rejected."""
        sample_loan_application["home_ownership"] = "UNKNOWN"

        with pytest.raises(ValidationError, match="home_ownership must be one of"):
            LoanApplication(**sample_loan_application)

    def test_invalid_verification_status(self, sample_loan_application: dict) -> None:
        """Test that invalid verification_status is rejected."""
        sample_loan_application["verification_status"] = "Unknown"

        with pytest.raises(ValidationError, match="verification_status must be one of"):
            LoanApplication(**sample_loan_application)

    def test_dti_bounds(self, sample_loan_application: dict) -> None:
        """Test DTI must be within bounds."""
        sample_loan_application["dti"] = 150

        with pytest.raises(ValidationError):
            LoanApplication(**sample_loan_application)

    def test_model_dump(self, sample_loan_application: dict) -> None:
        """Test that model can be serialized to dict."""
        app = LoanApplication(**sample_loan_application)
        data = app.model_dump()

        assert isinstance(data, dict)
        assert data["loan_amnt"] == 15000.0


class TestRiskPrediction:
    """Tests for RiskPrediction schema."""

    def test_valid_prediction(self) -> None:
        """Test creating a valid risk prediction."""
        pred = RiskPrediction(
            risk_score=0.35,
            risk_tier=RiskTier.MEDIUM,
            recommendation=Recommendation.REVIEW,
            model_version="v1.0.0",
            explanation=None,
        )

        assert pred.risk_score == 0.35
        assert pred.risk_tier == RiskTier.MEDIUM

    def test_risk_score_bounds(self) -> None:
        """Test that risk score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            RiskPrediction(
                risk_score=1.5,
                risk_tier=RiskTier.HIGH,
                recommendation=Recommendation.DECLINE,
                model_version="v1.0.0",
            )

    def test_with_explanation(self) -> None:
        """Test prediction with explanation."""
        explanation = PredictionExplanation(
            base_value=0.15,
            top_contributors=[
                FeatureContribution(feature="dti", value=35.5, contribution=0.12),
                FeatureContribution(feature="grade", value="D", contribution=0.08),
            ],
        )

        pred = RiskPrediction(
            risk_score=0.35,
            risk_tier=RiskTier.MEDIUM,
            recommendation=Recommendation.REVIEW,
            model_version="v1.0.0",
            explanation=explanation,
        )

        assert pred.explanation is not None
        assert len(pred.explanation.top_contributors) == 2


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_batch_request(self, sample_loan_application: dict) -> None:
        """Test creating a valid batch request."""
        request = BatchPredictionRequest(
            applications=[LoanApplication(**sample_loan_application)],
            include_explanations=True,
        )

        assert len(request.applications) == 1
        assert request.include_explanations is True

    def test_empty_applications_rejected(self) -> None:
        """Test that empty applications list is rejected."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                applications=[],
                include_explanations=False,
            )

    def test_max_applications_limit(self, sample_loan_application: dict) -> None:
        """Test that exceeding max applications is rejected."""
        apps = [LoanApplication(**sample_loan_application)] * 101

        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                applications=apps,
                include_explanations=False,
            )


class TestEnums:
    """Tests for enum types."""

    def test_risk_tier_values(self) -> None:
        """Test RiskTier enum values."""
        assert RiskTier.LOW.value == "low"
        assert RiskTier.MEDIUM.value == "medium"
        assert RiskTier.HIGH.value == "high"
        assert RiskTier.VERY_HIGH.value == "very_high"

    def test_recommendation_values(self) -> None:
        """Test Recommendation enum values."""
        assert Recommendation.APPROVE.value == "approve"
        assert Recommendation.REVIEW.value == "review"
        assert Recommendation.ENHANCED_REVIEW.value == "enhanced_review"
        assert Recommendation.DECLINE.value == "decline"
