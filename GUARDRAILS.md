# Project Guardrails

Operational, statistical, and security boundaries enforced across `ab_test_analysis`.

## 1. Input & Schema Guardrails
- **Strict Pydantic v2 Models (`src/schemas.py`)**: All configuration (`ExperimentConfig`) and event rows (`UserEventRecord`) enforce `extra="forbid"` and `allow_inf_nan=False` on numeric fields.
- **Funnel & Revenue Invariants**: `UserEventRecord` enforces `viewed_landing >= clicked_cta >= started_checkout >= converted` and `revenue == 0.0` whenever `converted` is `False`.
- **Locale Allowlist (`src/i18n.py`)**: Only `en` and `id` locales are accepted; unsupported locale identifiers raise `ValueError`.
- **Bounded UI Inputs (`app.py`)**: Streamlit sidebar sliders/inputs enforce explicit numeric ranges and route through `ExperimentConfig` validation before execution; invalid inputs display generic localized error messages without stack traces.

## 2. Statistical & Experimentation Guardrails
- **Sample Ratio Mismatch (SRM) Gate (`src/frequentist.py`)**: A chi-square goodness-of-fit test (`alpha = 0.001`) runs before interpreting treatment effects. Any SRM failure forces `decision = "HOLD"` regardless of observed lift.
- **Pre-Specified Power & Peeking Prevention (`src/power.py`)**: Required sample size per variant and estimated test duration are computed a priori from baseline conversion rate, MDE, `alpha`, and target power (`0.80` default). Early stopping without alpha-spending is prohibited.
- **Multiple Testing Control (`src/frequentist.py`)**: Segment tests (`device` and `channel`) adjust raw p-values using Benjamini-Hochberg False Discovery Rate (`fdr_bh`) to control false positives across subgroups.
- **Dual-Paradigm Confirmation (`src/report.py`)**: `"SHIP"` requires SRM pass, Frequentist significance (`p < alpha`), Bayesian `P(Treatment > Control) >= 0.95`, and `P(Lift > ROPE) >= 0.90`.
- **Prior Sensitivity Check (`src/bayesian.py`)**: Every Bayesian evaluation reports results across Uniform `Beta(1, 1)`, Informative `Beta(10, 90)`, and Skeptical `Beta(20, 180)` priors.

## 3. Application & Container Security Guardrails
- **No Unsafe Deserialization or Execution**: Zero `pickle`, `marshal`, `eval`, `exec`, or `shell=True` subprocess calls. Serialization uses JSON (`model_dump_json`) and CSV only.
- **XSS Mitigation (`app.py`, `src/report.py`)**: Dynamic values rendered into HTML/ReportLab blocks are escaped with `html.escape()` / `xml.sax.saxutils.escape()`.
- **Headless Rendering (`src/visualizations.py`)**: Matplotlib explicitly uses the non-interactive `Agg` backend.
- **Hardened Container (`Dockerfile`)**: Multi-stage Python 3.12-slim build, non-root `appuser`, pinned runtime dependencies, built-in `HEALTHCHECK`, and pre-generated report artifacts during image build.
- **Automated CI Gates (`.github/workflows/guardrails.yml`)**: Least-privilege `contents: read` permissions running `ruff`, `mypy`, `bandit`, `pip-audit`, and `pytest`.
