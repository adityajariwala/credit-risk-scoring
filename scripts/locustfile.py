"""
Load testing script using Locust.

Run with: locust -f scripts/locustfile.py --host=http://localhost:8000

Web UI will be available at http://localhost:8089
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


class CreditRiskUser(HttpUser):
    """
    Simulated user making requests to the credit risk scoring API.

    Includes realistic traffic patterns:
    - Health checks (low frequency)
    - Single predictions (high frequency)
    - Batch predictions (medium frequency)
    - Model info (low frequency)
    """

    # Wait between 0.5 and 2 seconds between requests
    wait_time = between(0.5, 2)

    def on_start(self) -> None:
        """Called when a simulated user starts."""
        # Verify the service is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("Service not healthy")

    @task(1)
    def health_check(self) -> None:
        """Health check endpoint - low frequency."""
        self.client.get("/health")

    @task(10)
    def single_prediction(self) -> None:
        """Single prediction - most common request type."""
        application = self._generate_application()
        self.client.post(
            "/predict",
            json=application,
            params={"include_explanation": random.choice([True, False])},
        )

    @task(3)
    def single_prediction_no_explanation(self) -> None:
        """Single prediction without explanation - faster."""
        application = self._generate_application()
        self.client.post(
            "/predict",
            json=application,
            params={"include_explanation": False},
        )

    @task(2)
    def batch_prediction_small(self) -> None:
        """Small batch prediction (5 applications)."""
        applications = [self._generate_application() for _ in range(5)]
        self.client.post(
            "/predict/batch",
            json={
                "applications": applications,
                "include_explanations": False,
            },
        )

    @task(1)
    def batch_prediction_large(self) -> None:
        """Large batch prediction (50 applications)."""
        applications = [self._generate_application() for _ in range(50)]
        self.client.post(
            "/predict/batch",
            json={
                "applications": applications,
                "include_explanations": False,
            },
        )

    @task(1)
    def model_info(self) -> None:
        """Model info endpoint - low frequency."""
        self.client.get("/model/info")

    def _generate_application(self) -> dict:
        """Generate a random but valid loan application."""
        grades = ["A", "B", "C", "D", "E", "F", "G"]
        grade = random.choices(grades, weights=[15, 25, 25, 15, 10, 7, 3])[0]

        return {
            "loan_amnt": round(random.uniform(1000, 40000), 2),
            "annual_inc": round(random.uniform(20000, 300000), 2),
            "dti": round(random.uniform(0, 50), 2),
            "open_acc": random.randint(1, 30),
            "revol_bal": round(random.uniform(0, 50000), 2),
            "revol_util": round(random.uniform(0, 100), 2),
            "total_acc": random.randint(1, 50),
            "int_rate": round(random.uniform(5, 25), 2),
            "installment": round(random.uniform(50, 1500), 2),
            "term": random.choice(["36 months", "60 months"]),
            "grade": grade,
            "home_ownership": random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"]),
            "verification_status": random.choice(["Verified", "Source Verified", "Not Verified"]),
            "purpose": random.choice(
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
                ]
            ),
        }


class HighThroughputUser(HttpUser):
    """
    High-throughput user for stress testing.

    Simulates automated systems making rapid requests.
    """

    wait_time = between(0.1, 0.3)
    weight = 1  # Lower weight than regular users

    @task
    def rapid_predictions(self) -> None:
        """Rapid single predictions without explanations."""
        application = {
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

        self.client.post(
            "/predict",
            json=application,
            params={"include_explanation": False},
        )
