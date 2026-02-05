"""
FastAPI inference service for credit risk scoring.

This module provides the HTTP API for real-time risk predictions,
including health checks, single predictions, batch predictions,
and model information endpoints.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.inference.model_service import ModelService
from src.inference.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthStatus,
    LoanApplication,
    ModelInfo,
    RiskPrediction,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions",
    ["risk_tier", "recommendation"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Global model service instance
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    global model_service

    # Startup
    logger.info("starting_application")

    config_path = Path("config/config.yaml")
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning("config_not_found", path=str(config_path))
        config = {}

    model_dir = Path(config.get("paths", {}).get("model_dir", "models"))

    if model_dir.exists() and (model_dir / "latest").exists():
        model_service = ModelService(model_dir=model_dir, config=config)
        try:
            model_service.load("latest")
            logger.info("model_loaded_on_startup", version=model_service.version)
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            model_service = None
    else:
        logger.warning("no_model_found", model_dir=str(model_dir))
        model_service = None

    yield

    # Shutdown
    logger.info("shutting_down_application")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk prediction service with ML model inference and SHAP explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    """Log all HTTP requests with timing."""
    start_time = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        elapsed_ms=round(elapsed_ms, 2),
    )

    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    return response


@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check() -> HealthStatus:
    """
    Health check endpoint for load balancers and orchestration.

    Returns service status and model availability.
    """
    return HealthStatus(
        status="healthy" if model_service and model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded if model_service else False,
        model_version=model_service.version if model_service else None,
    )


@app.get("/ready", tags=["System"])
async def readiness_check() -> JSONResponse:
    """
    Readiness check endpoint.

    Returns 200 only when the service is ready to accept traffic.
    """
    if model_service and model_service.is_loaded:
        return JSONResponse(content={"ready": True}, status_code=status.HTTP_200_OK)

    return JSONResponse(
        content={"ready": False, "reason": "Model not loaded"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/metrics", tags=["System"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info() -> ModelInfo:
    """
    Get information about the currently loaded model.

    Returns model metadata including version, features, and training metrics.
    """
    if not model_service or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    info = model_service.get_info()
    return ModelInfo(**info)


@app.post("/predict", response_model=RiskPrediction, tags=["Prediction"])
async def predict(
    application: LoanApplication,
    include_explanation: bool = True,
) -> RiskPrediction:
    """
    Generate risk prediction for a single loan application.

    Args:
        application: Loan application details
        include_explanation: Whether to include SHAP-based explanation

    Returns:
        Risk prediction with score, tier, recommendation, and optional explanation
    """
    if not model_service or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start_time = time.perf_counter()

    try:
        prediction = model_service.predict(application, include_explanation)

        # Record metrics
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_COUNTER.labels(
            risk_tier=prediction.risk_tier.value,
            recommendation=prediction.recommendation.value,
        ).inc()

        return prediction

    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate risk predictions for multiple loan applications.

    Optimized for batch processing with vectorized operations.
    Maximum 100 applications per request.

    Args:
        request: Batch prediction request with list of applications

    Returns:
        Batch prediction response with all predictions and processing time
    """
    if not model_service or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start_time = time.perf_counter()

    try:
        predictions = model_service.predict_batch(
            request.applications,
            request.include_explanations,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        for pred in predictions:
            PREDICTION_COUNTER.labels(
                risk_tier=pred.risk_tier.value,
                recommendation=pred.recommendation.value,
            ).inc()

        return BatchPredictionResponse(
            predictions=predictions,
            processing_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error("batch_prediction_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def main() -> None:
    """Entry point for running the server."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        config = load_config(str(config_path))
        server_config = config.get("server", {})
    else:
        server_config = {}

    uvicorn.run(
        "src.inference.app:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        workers=server_config.get("workers", 1),
        reload=False,
    )


if __name__ == "__main__":
    main()
