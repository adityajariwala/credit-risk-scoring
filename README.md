# Credit Risk Scoring System

A "sandbox" end-to-end machine learning system for real-time credit risk prediction.

## Features

- **Real-time Inference API**: FastAPI service with sub-100ms latency
- **SHAP Explanations**: Interpretable predictions with feature contributions
- **Feature Pipeline Parity**: Identical transformations for training and inference
- **Model Versioning**: Config-driven deployment with version management
- **Comprehensive Testing**: Unit, integration, and load tests
- **CI/CD Pipeline**: GitHub Actions for automated testing and validation
- **Docker Support**: Multi-stage build for optimized production images
- **Observability**: Prometheus metrics and structured logging

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Credit Risk Scoring System                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Training   │    │   Feature    │    │    Model Registry    │  │
│  │   Pipeline   │───▶│   Pipeline   │───▶│  (versioned models)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Offline    │    │   Online     │    │      FastAPI         │  │
│  │   Training   │    │   Inference  │◀───│   Inference Service  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                     │               │
│                                                     ▼               │
│                                          ┌──────────────────────┐  │
│                                          │  SHAP Explanations   │  │
│                                          │  + Risk Tiers        │  │
│                                          └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
credit-risk-scoring/
├── src/
│   ├── features/           # Feature engineering pipeline
│   │   ├── transforms.py   # Individual feature transformations
│   │   └── pipeline.py     # Unified feature pipeline
│   ├── training/
│   │   └── train.py        # Model training pipeline
│   ├── inference/
│   │   ├── app.py          # FastAPI application
│   │   ├── model_service.py # Model loading and prediction
│   │   └── schemas.py      # Pydantic request/response models
│   └── evaluation/
│       └── metrics.py      # Evaluation metrics
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── config/
│   └── config.yaml         # Central configuration
├── scripts/
│   └── locustfile.py       # Load testing
├── .github/workflows/
│   └── ci.yaml             # CI/CD pipeline
├── Dockerfile              # Container definition
├── docker-compose.yaml     # Multi-service orchestration
└── pyproject.toml          # Project configuration
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/adityajariwala/credit-risk-scoring
cd credit-risk-scoring

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### Train a Model

```bash
# Train with sample data (automatically generated)
python -m src.training.train --create-sample-data

# Train with custom data
python -m src.training.train --data path/to/data.csv
```

### Run the API Server

```bash
# Start the server
uvicorn src.inference.app:app --reload

# Or use the CLI entry point
python -m src.inference.app
```

### Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "annual_inc": 75000,
    "dti": 18.5,
    "open_acc": 8,
    "revol_bal": 12500,
    "revol_util": 45.2,
    "total_acc": 20,
    "int_rate": 12.5,
    "installment": 450,
    "term": "36 months",
    "grade": "B",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "purpose": "debt_consolidation"
  }'
```

### Run with Docker

```bash
# Build and run
docker-compose up --build

# With monitoring stack
docker-compose --profile monitoring up --build
```

## API Endpoints

| Endpoint         | Method | Description                   |
|------------------|--------|-------------------------------|
| `/health`        | GET    | Health check                  |
| `/ready`         | GET    | Readiness probe               |
| `/metrics`       | GET    | Prometheus metrics            |
| `/predict`       | POST   | Single prediction             |
| `/predict/batch` | POST   | Batch predictions (up to 100) |
| `/model/info`    | GET    | Model metadata                |

### Example Response

```json
{
  "risk_score": 0.35,
  "risk_tier": "medium",
  "recommendation": "review",
  "model_version": "v1.0.0",
  "explanation": {
    "base_value": 0.15,
    "top_contributors": [
      {"feature": "dti", "value": 18.5, "contribution": 0.08},
      {"feature": "grade", "value": "B", "contribution": 0.05}
    ]
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit -v
pytest tests/integration -v
```

## Load Testing

```bash
# Install locust
pip install locust

# Run load test (Web UI at http://localhost:8089)
locust -f scripts/locustfile.py --host=http://localhost:8000
```

## Configuration

All configuration is centralized in `config/config.yaml`:

```yaml
model:
  name: "credit_risk_lgbm"
  version: "v1.0.0"
  params:
    n_estimators: 200
    learning_rate: 0.05
    # ...

features:
  numeric:
    - name: "loan_amnt"
      transform: "log1p"
  categorical:
    - name: "grade"
      encoding: "ordinal"
  derived:
    - name: "loan_to_income_ratio"
      formula: "loan_amnt / (annual_inc + 1)"

inference:
  risk_tiers:
    - name: "low"
      max_score: 0.2
      recommendation: "approve"
    # ...
```

## Key Design Decisions

### Feature Pipeline Parity

The same `FeaturePipeline` class is used for both training and inference, ensuring transformations are identical. The pipeline is serialized with the model and loaded during inference.

```python
# Training
pipeline = FeaturePipeline(config)
X_train = pipeline.fit_transform(df, target)
pipeline.save("models/v1.0.0/pipeline.json")

# Inference
pipeline = FeaturePipeline.load("models/v1.0.0/pipeline.json")
features = pipeline.transform_single(request_data)
```

### Model Versioning

Models are stored in versioned directories with a `latest` symlink for easy deployment:

```
models/
├── v1.0.0/
│   ├── model.joblib
│   ├── pipeline.json
│   └── metadata.json
├── v1.1.0/
│   └── ...
└── latest -> v1.1.0
```

### SHAP Explanations

Every prediction can include SHAP-based feature contributions, making the model interpretable:

```python
# Top features driving this prediction:
# dti: +0.12 (high debt-to-income increases risk)
# grade: +0.08 (lower grade increases risk)
# annual_inc: -0.05 (higher income decreases risk)
```

## Performance

Typical latency (on 4-core CPU):
- Single prediction: ~20-50ms
- Single prediction with SHAP: ~50-100ms
- Batch prediction (100 samples): ~200-500ms

## Monitoring

The service exposes Prometheus metrics at `/metrics`:

- `predictions_total` - Counter by risk tier and recommendation
- `prediction_latency_seconds` - Histogram of prediction latency
- `http_requests_total` - Counter by method, endpoint, status

## License

MIT
