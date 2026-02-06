export interface LoanApplication {
    loan_amnt: number;
    annual_inc: number;
    dti: number;
    open_acc: number;
    revol_bal: number;
    revol_util: number;
    total_acc: number;
    int_rate: number;
    installment: number;
    term: string;
    grade: string;
    home_ownership: string;
    verification_status: string;
    purpose: string;
}

export interface FeatureContribution {
    feature: string;
    value: number | string;
    contribution: number;
}

export interface PredictionExplanation {
    base_value: number;
    top_contributors: FeatureContribution[];
}

export type RiskTier = "low" | "medium" | "high" | "very_high";
export type Recommendation = "approve" | "review" | "enhanced_review" | "decline";

export interface RiskPrediction {
    risk_score: number;
    risk_tier: RiskTier;
    recommendation: Recommendation;
    model_version: string;
    explanation: PredictionExplanation | null;
}

export interface HealthStatus {
    status: string;
    model_loaded: boolean;
    model_version: string | null;
}

export const DEFAULT_APPLICATION: LoanApplication = {
    loan_amnt: 15000,
    annual_inc: 75000,
    dti: 18.5,
    open_acc: 8,
    revol_bal: 12500,
    revol_util: 45.2,
    total_acc: 20,
    int_rate: 12.5,
    installment: 450,
    term: "36 months",
    grade: "B",
    home_ownership: "MORTGAGE",
    verification_status: "Verified",
    purpose: "debt_consolidation",
};
