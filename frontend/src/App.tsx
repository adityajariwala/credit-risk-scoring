import {useEffect, useState} from "react";
import {
    Alert,
    AppBar,
    Box,
    Button,
    Chip,
    CircularProgress,
    Container,
    FormControl,
    Grid,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    TextField,
    Toolbar,
    Typography,
} from "@mui/material";
import {createTheme, ThemeProvider} from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import type {HealthStatus, LoanApplication, RiskPrediction, RiskTier,} from "./types";
import {DEFAULT_APPLICATION} from "./types";
import {fetchHealth, predict} from "./api";

const theme = createTheme({
    palette: {
        mode: "light",
        primary: {main: "#1565c0"},
        success: {main: "#2e7d32"},
        warning: {main: "#ed6c02"},
        error: {main: "#d32f2f"},
    },
});

const TIER_COLOR: Record<RiskTier, "success" | "warning" | "error"> = {
    low: "success",
    medium: "warning",
    high: "error",
    very_high: "error",
};

const TIER_LABEL: Record<RiskTier, string> = {
    low: "Low Risk",
    medium: "Medium Risk",
    high: "High Risk",
    very_high: "Very High Risk",
};

const PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "moving",
    "vacation",
    "other",
];

export default function App() {
    const [form, setForm] = useState<LoanApplication>(DEFAULT_APPLICATION);
    const [result, setResult] = useState<RiskPrediction | null>(null);
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchHealth().then(setHealth).catch(() => setHealth(null));
    }, []);

    const handleChange = (field: keyof LoanApplication, value: string) => {
        setForm((prev) => ({
            ...prev,
            [field]: [
                "term",
                "grade",
                "home_ownership",
                "verification_status",
                "purpose",
            ].includes(field)
                ? value
                : Number(value),
        }));
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const prediction = await predict(form);
            setResult(prediction);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Prediction failed");
        } finally {
            setLoading(false);
        }
    };

    const modelReady = health?.model_loaded ?? false;

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <AppBar position="static" elevation={1}>
                <Toolbar>
                    <Typography variant="h6" sx={{flexGrow: 1}}>
                        Credit Risk Scoring
                    </Typography>
                    <Chip
                        label={
                            health
                                ? `Model ${health.model_version ?? "unknown"} — ${health.status}`
                                : "API unavailable"
                        }
                        color={modelReady ? "success" : "error"}
                        size="small"
                        variant="outlined"
                        sx={{color: "white", borderColor: "rgba(255,255,255,0.5)"}}
                    />
                </Toolbar>
            </AppBar>

            <Container maxWidth="lg" sx={{mt: 4, mb: 4}}>
                <Grid container spacing={3}>
                    {/* --- Form Panel --- */}
                    <Grid size={{xs: 12, md: 5}}>
                        <Paper sx={{p: 3}}>
                            <Typography variant="h6" gutterBottom>
                                Loan Application
                            </Typography>

                            <Grid container spacing={2}>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Loan Amount ($)"
                                        type="number"
                                        size="small"
                                        value={form.loan_amnt}
                                        onChange={(e) => handleChange("loan_amnt", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Annual Income ($)"
                                        type="number"
                                        size="small"
                                        value={form.annual_inc}
                                        onChange={(e) => handleChange("annual_inc", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="DTI (%)"
                                        type="number"
                                        size="small"
                                        value={form.dti}
                                        onChange={(e) => handleChange("dti", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Interest Rate (%)"
                                        type="number"
                                        size="small"
                                        value={form.int_rate}
                                        onChange={(e) => handleChange("int_rate", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Installment ($/mo)"
                                        type="number"
                                        size="small"
                                        value={form.installment}
                                        onChange={(e) =>
                                            handleChange("installment", e.target.value)
                                        }
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Revolving Balance ($)"
                                        type="number"
                                        size="small"
                                        value={form.revol_bal}
                                        onChange={(e) => handleChange("revol_bal", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Revolving Util (%)"
                                        type="number"
                                        size="small"
                                        value={form.revol_util}
                                        onChange={(e) => handleChange("revol_util", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Open Accounts"
                                        type="number"
                                        size="small"
                                        value={form.open_acc}
                                        onChange={(e) => handleChange("open_acc", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <TextField
                                        fullWidth
                                        label="Total Accounts"
                                        type="number"
                                        size="small"
                                        value={form.total_acc}
                                        onChange={(e) => handleChange("total_acc", e.target.value)}
                                    />
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Term</InputLabel>
                                        <Select
                                            value={form.term}
                                            label="Term"
                                            onChange={(e) => handleChange("term", e.target.value)}
                                        >
                                            <MenuItem value="36 months">36 months</MenuItem>
                                            <MenuItem value="60 months">60 months</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Grade</InputLabel>
                                        <Select
                                            value={form.grade}
                                            label="Grade"
                                            onChange={(e) => handleChange("grade", e.target.value)}
                                        >
                                            {["A", "B", "C", "D", "E", "F", "G"].map((g) => (
                                                <MenuItem key={g} value={g}>
                                                    {g}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Home Ownership</InputLabel>
                                        <Select
                                            value={form.home_ownership}
                                            label="Home Ownership"
                                            onChange={(e) =>
                                                handleChange("home_ownership", e.target.value)
                                            }
                                        >
                                            {["RENT", "OWN", "MORTGAGE", "OTHER"].map((v) => (
                                                <MenuItem key={v} value={v}>
                                                    {v}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Verification</InputLabel>
                                        <Select
                                            value={form.verification_status}
                                            label="Verification"
                                            onChange={(e) =>
                                                handleChange("verification_status", e.target.value)
                                            }
                                        >
                                            {["Verified", "Source Verified", "Not Verified"].map(
                                                (v) => (
                                                    <MenuItem key={v} value={v}>
                                                        {v}
                                                    </MenuItem>
                                                )
                                            )}
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid size={{xs: 6}}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Purpose</InputLabel>
                                        <Select
                                            value={form.purpose}
                                            label="Purpose"
                                            onChange={(e) => handleChange("purpose", e.target.value)}
                                        >
                                            {PURPOSES.map((p) => (
                                                <MenuItem key={p} value={p}>
                                                    {p.replace(/_/g, " ")}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>

                            <Button
                                variant="contained"
                                fullWidth
                                sx={{mt: 3}}
                                onClick={handleSubmit}
                                disabled={loading || !modelReady}
                            >
                                {loading ? (
                                    <CircularProgress size={24} color="inherit"/>
                                ) : (
                                    "Score Application"
                                )}
                            </Button>

                            {!modelReady && (
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{mt: 1, display: "block"}}
                                >
                                    Start the API server to enable scoring.
                                </Typography>
                            )}
                        </Paper>
                    </Grid>

                    {/* --- Results Panel --- */}
                    <Grid size={{xs: 12, md: 7}}>
                        {error && (
                            <Alert severity="error" sx={{mb: 2}}>
                                {error}
                            </Alert>
                        )}

                        {result && (
                            <>
                                {/* Risk Scorecard */}
                                <Paper sx={{p: 3, mb: 3}}>
                                    <Grid container spacing={2} alignItems="center">
                                        <Grid size={{xs: 12, sm: 4}} sx={{textAlign: "center"}}>
                                            <Typography variant="overline" color="text.secondary">
                                                Risk Score
                                            </Typography>
                                            <Typography
                                                variant="h2"
                                                fontWeight="bold"
                                                color={`${TIER_COLOR[result.risk_tier]}.main`}
                                            >
                                                {(result.risk_score * 100).toFixed(1)}%
                                            </Typography>
                                        </Grid>
                                        <Grid size={{xs: 6, sm: 4}} sx={{textAlign: "center"}}>
                                            <Typography variant="overline" color="text.secondary">
                                                Risk Tier
                                            </Typography>
                                            <Box>
                                                <Chip
                                                    label={TIER_LABEL[result.risk_tier]}
                                                    color={TIER_COLOR[result.risk_tier]}
                                                    sx={{mt: 0.5}}
                                                />
                                            </Box>
                                        </Grid>
                                        <Grid size={{xs: 6, sm: 4}} sx={{textAlign: "center"}}>
                                            <Typography variant="overline" color="text.secondary">
                                                Recommendation
                                            </Typography>
                                            <Typography
                                                variant="h6"
                                                fontWeight="medium"
                                                sx={{textTransform: "uppercase", mt: 0.5}}
                                            >
                                                {result.recommendation.replace(/_/g, " ")}
                                            </Typography>
                                        </Grid>
                                    </Grid>
                                </Paper>

                                {/* SHAP Explanation */}
                                {result.explanation && (
                                    <Paper sx={{p: 3}}>
                                        <Typography variant="h6" gutterBottom>
                                            Feature Contributions (SHAP)
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary" sx={{mb: 2}}>
                                            Bars to the right increase risk (red). Bars to the left decrease risk
                                            (green). Values show contribution to the final risk score.
                                        </Typography>

                                        {(() => {
                                            const contributors = result.explanation!.top_contributors;

                                            // Optional: drop near-zero noise
                                            const filtered = contributors.filter(
                                                (c) => Math.abs(c.contribution) > 0.0005
                                            );

                                            const maxAbs = Math.max(
                                                ...filtered.map((c) => Math.abs(c.contribution)),
                                                1e-6
                                            );

                                            return filtered.map((c) => {
                                                const isRisk = c.contribution > 0;
                                                const barWidth = (Math.abs(c.contribution) / maxAbs) * 50;
                                                const pretty = c.feature.replace(/_/g, " ");

                                                return (
                                                    <Box
                                                        key={c.feature}
                                                        sx={{
                                                            display: "flex",
                                                            alignItems: "center",
                                                            mb: 1,
                                                            height: 28,
                                                        }}
                                                    >
                                                        {/* Feature label */}
                                                        <Typography
                                                            variant="body2"
                                                            sx={{
                                                                width: 150,
                                                                textAlign: "right",
                                                                pr: 2,
                                                                fontWeight: 500,
                                                                whiteSpace: "nowrap",
                                                                overflow: "hidden",
                                                                textOverflow: "ellipsis",
                                                            }}
                                                        >
                                                            {pretty}
                                                        </Typography>

                                                        {/* Bar area */}
                                                        <Box
                                                            sx={{
                                                                position: "relative",
                                                                flexGrow: 1,
                                                                height: 20,
                                                                bgcolor: "transparent",
                                                                border: "1px solid",
                                                                borderColor: "grey.200",
                                                                borderRadius: 1,
                                                            }}
                                                        >
                                                            {/* Zero line */}
                                                            <Box
                                                                sx={{
                                                                    position: "absolute",
                                                                    left: "50%",
                                                                    top: 0,
                                                                    bottom: 0,
                                                                    width: 1,
                                                                    bgcolor: "grey.400",
                                                                }}
                                                            />

                                                            {/* Contribution bar */}
                                                            <Box
                                                                sx={{
                                                                    position: "absolute",
                                                                    left: "50%",
                                                                    height: "100%",
                                                                    width: `${barWidth}%`,
                                                                    bgcolor: isRisk ? "error.main" : "success.main",
                                                                    transform: isRisk ? "none" : "translateX(-100%)",
                                                                    borderRadius: 1,
                                                                }}
                                                            />
                                                        </Box>

                                                        {/* Value */}
                                                        <Typography
                                                            variant="caption"
                                                            sx={{
                                                                width: 80,
                                                                pl: 1.5,
                                                                fontFamily: "monospace",
                                                                fontWeight: 600,
                                                                color: isRisk ? "error.main" : "success.main",
                                                            }}
                                                        >
                                                            {`${isRisk ? "+" : "−"}${Math.abs(
                                                                c.contribution * 100
                                                            ).toFixed(1)}%`}
                                                        </Typography>
                                                    </Box>
                                                );
                                            });
                                        })()}
                                    </Paper>
                                )}
                            </>
                        )}

                        {!result && !error && (
                            <Paper
                                sx={{
                                    p: 6,
                                    textAlign: "center",
                                    color: "text.secondary",
                                }}
                            >
                                <Typography variant="h6" gutterBottom>
                                    Submit a loan application to see the risk assessment
                                </Typography>
                                <Typography variant="body2">
                                    The model returns a probability of default, risk tier, lending
                                    recommendation, and SHAP-based feature explanations.
                                </Typography>
                            </Paper>
                        )}
                    </Grid>
                </Grid>
            </Container>
        </ThemeProvider>
    );
}
