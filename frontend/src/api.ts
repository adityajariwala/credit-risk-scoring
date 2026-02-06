import type {HealthStatus, LoanApplication, RiskPrediction} from "./types";

const BASE = "/api";

export async function fetchHealth(): Promise<HealthStatus> {
    const res = await fetch(`${BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return res.json();
}

export async function predict(
    application: LoanApplication
): Promise<RiskPrediction> {
    const res = await fetch(`${BASE}/predict?include_explanation=true`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(application),
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({detail: res.statusText}));
        throw new Error(detail.detail || `Prediction failed: ${res.status}`);
    }
    return res.json();
}
