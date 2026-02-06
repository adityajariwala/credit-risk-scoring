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
                                        <Typography
                                            variant="body2"
                                            color="text.secondary"
                                            sx={{mb: 2}}
                                        >
                                            Each bar shows how a feature pushes the risk score up
                                            (red, increases risk) or down (green, decreases risk).
                                        </Typography>
                                        {(() => {
                                            const contributors =
                                                result.explanation!.top_contributors;
                                            const maxAbs = Math.max(
                                                ...contributors.map((x) => Math.abs(x.contribution))
                                            );
                                            return contributors.map((c) => {
                                                const barPct =
                                                    maxAbs > 0
                                                        ? (Math.abs(c.contribution) / maxAbs) * 100
                                                        : 0;
                                                const isRisk = c.contribution > 0;
                                                const pretty = c.feature.replace(/_/g, " ");
                                                return (
                                                    <Box
                                                        key={pretty}
                                                        sx={{
                                                            display: "flex",
                                                            alignItems: "center",
                                                            mb: 0.75,
                                                            height: 28,
                                                        }}
                                                    >
                                                        <Typography
                                                            variant="body2"
                                                            sx={{
                                                                width: 140,
                                                                flexShrink: 0,
                                                                textAlign: "right",
                                                                pr: 1.5,
                                                                fontFamily: "inherit",
                                                                fontWeight: 500,
                                                                fontSize: "0.8rem",
                                                                overflow: "hidden",
                                                                textOverflow: "ellipsis",
                                                                whiteSpace: "nowrap",
                                                            }}
                                                        >
                                                            {pretty}
                                                        </Typography>
                                                        <Box
                                                            sx={{
                                                                flexGrow: 1,
                                                                display: "flex",
                                                                alignItems: "center",
                                                                position: "relative",
                                                                height: "100%",
                                                            }}
                                                        >
                                                            {/* Left half (safe / green) */}
                                                            <Box
                                                                sx={{
                                                                    width: "50%",
                                                                    height: "100%",
                                                                    display: "flex",
                                                                    justifyContent: "flex-end",
                                                                    alignItems: "center",
                                                                    pr: "1px",
                                                                }}
                                                            >
                                                                {!isRisk && (
                                                                    <Box
                                                                        sx={{
                                                                            width: `${barPct}%`,
                                                                            height: 20,
                                                                            bgcolor: "#2e7d32",
                                                                            borderRadius: "4px 0 0 4px",
                                                                        }}
                                                                    />
                                                                )}
                                                            </Box>
                                                            {/* Center divider */}
                                                            <Box
                                                                sx={{
                                                                    width: 2,
                                                                    height: "100%",
                                                                    bgcolor: "grey.300",
                                                                    opacity: 0.6,
                                                                    flexShrink: 0,
                                                                }}
                                                            />
                                                            {/* Right half (risk / red) */}
                                                            <Box
                                                                sx={{
                                                                    width: "50%",
                                                                    height: "100%",
                                                                    display: "flex",
                                                                    justifyContent: "flex-start",
                                                                    alignItems: "center",
                                                                    pl: "1px",
                                                                }}
                                                            >
                                                                {isRisk && (
                                                                    <Box
                                                                        sx={{
                                                                            width: `${barPct}%`,
                                                                            height: 20,
                                                                            bgcolor: "#d32f2f",
                                                                            borderRadius: "0 4px 4px 0",
                                                                        }}
                                                                    />
                                                                )}
                                                            </Box>
                                                        </Box>
                                                        <Typography
                                                            variant="caption"
                                                            sx={{
                                                                width: 90,
                                                                pl: 1,
                                                                flexShrink: 0,
                                                                fontFamily: "monospace",
                                                                fontWeight: 600,
                                                                color: isRisk ? "#d32f2f" : "#2e7d32",
                                                            }}
                                                        >
                                                            {isRisk ? "+" : ""}
                                                            {`${c.contribution > 0 ? "+" : "−"}${Math.abs(c.contribution * 100).toFixed(1)}%`}
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
