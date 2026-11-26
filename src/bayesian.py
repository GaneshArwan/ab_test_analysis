"""Bayesian Beta-Binomial conversion inference, HDI, ROPE, and prior sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.schemas import BayesianResult, ExperimentConfig, PriorSensitivityResult

_SENSITIVITY_PRIORS: tuple[tuple[str, float, float], ...] = (
    ("Weak Uniform Beta(1, 1)", 1.0, 1.0),
    ("Informative Beta(10, 90)", 10.0, 90.0),
    ("Skeptical Beta(20, 180)", 20.0, 180.0),
)


@dataclass(frozen=True)
class _PosteriorMetrics:
    control_post_alpha: float
    control_post_beta: float
    treatment_post_alpha: float
    treatment_post_beta: float
    control_post_mean: float
    treatment_post_mean: float
    prob_beats: float
    loss_treatment: float
    loss_control: float
    abs_diff_draws: np.ndarray
    abs_mean: float
    abs_hdi_low: float
    abs_hdi_high: float
    rel_mean: float
    rel_hdi_low: float
    rel_hdi_high: float


def compute_hdi(
    samples: np.ndarray,
    credible_mass: float = 0.95,
) -> tuple[float, float]:
    """Compute shortest Highest Density Interval (HDI) from Monte Carlo samples."""
    arr = np.asarray(samples, dtype=float).ravel()
    n = arr.size
    if n < 2:
        raise ValueError("At least 2 samples are required to compute an HDI.")
    if not (0.0 < credible_mass < 1.0):
        raise ValueError("credible_mass must lie strictly in (0, 1).")

    sorted_samples = np.sort(arr)
    interval_idx = max(1, min(n - 1, int(np.floor(credible_mass * n))))
    n_intervals = n - interval_idx
    widths = sorted_samples[interval_idx:] - sorted_samples[:n_intervals]
    min_idx = int(np.argmin(widths))
    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + interval_idx])


def _posterior_metrics(
    control_n: int,
    control_conv: int,
    treatment_n: int,
    treatment_conv: int,
    prior_alpha: float,
    prior_beta: float,
    rng: np.random.Generator,
    n_draws: int,
) -> _PosteriorMetrics:
    c_alpha = prior_alpha + control_conv
    c_beta = prior_beta + (control_n - control_conv)
    t_alpha = prior_alpha + treatment_conv
    t_beta = prior_beta + (treatment_n - treatment_conv)

    p_c = rng.beta(c_alpha, c_beta, size=n_draws)
    p_t = rng.beta(t_alpha, t_beta, size=n_draws)
    abs_diff = p_t - p_c
    rel_lift = np.where(p_c > 0.0, abs_diff / p_c, 0.0)

    abs_low, abs_high = compute_hdi(abs_diff, credible_mass=0.95)
    rel_low, rel_high = compute_hdi(rel_lift, credible_mass=0.95)

    return _PosteriorMetrics(
        control_post_alpha=c_alpha,
        control_post_beta=c_beta,
        treatment_post_alpha=t_alpha,
        treatment_post_beta=t_beta,
        control_post_mean=c_alpha / (c_alpha + c_beta),
        treatment_post_mean=t_alpha / (t_alpha + t_beta),
        prob_beats=float(np.mean(p_t > p_c)),
        loss_treatment=float(np.mean(np.maximum(p_c - p_t, 0.0))),
        loss_control=float(np.mean(np.maximum(p_t - p_c, 0.0))),
        abs_diff_draws=abs_diff,
        abs_mean=float(np.mean(abs_diff)),
        abs_hdi_low=abs_low,
        abs_hdi_high=abs_high,
        rel_mean=float(np.mean(rel_lift)),
        rel_hdi_low=rel_low,
        rel_hdi_high=rel_high,
    )


def run_bayesian_analysis(
    control_n: int,
    control_conv: int,
    treatment_n: int,
    treatment_conv: int,
    config: ExperimentConfig,
    n_draws: int = 100000,
) -> BayesianResult:
    """Run conjugate Beta-Binomial Bayesian analysis with ROPE and prior sensitivity."""
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("control_n and treatment_n must be > 0.")
    if not (0 <= control_conv <= control_n and 0 <= treatment_conv <= treatment_n):
        raise ValueError("Conversions must be within [0, arm_n].")
    if n_draws < 100:
        raise ValueError("n_draws must be >= 100.")

    rng = np.random.default_rng(config.seed)
    primary = _posterior_metrics(
        control_n=control_n,
        control_conv=control_conv,
        treatment_n=treatment_n,
        treatment_conv=treatment_conv,
        prior_alpha=config.prior_alpha,
        prior_beta=config.prior_beta,
        rng=rng,
        n_draws=n_draws,
    )

    rope_margin = config.rope_margin
    prob_in_rope = float(np.mean(np.abs(primary.abs_diff_draws) <= rope_margin))
    prob_above_rope = float(np.mean(primary.abs_diff_draws > rope_margin))
    prob_below_rope = float(np.mean(primary.abs_diff_draws < -rope_margin))

    sensitivity: list[PriorSensitivityResult] = []
    for name, p_a, p_b in _SENSITIVITY_PRIORS:
        sens_rng = np.random.default_rng(config.seed)
        sens = _posterior_metrics(
            control_n=control_n,
            control_conv=control_conv,
            treatment_n=treatment_n,
            treatment_conv=treatment_conv,
            prior_alpha=p_a,
            prior_beta=p_b,
            rng=sens_rng,
            n_draws=n_draws,
        )
        sensitivity.append(
            PriorSensitivityResult(
                prior_name=name,
                prior_alpha=p_a,
                prior_beta=p_b,
                prob_treatment_beats_control=sens.prob_beats,
                expected_loss_treatment=sens.loss_treatment,
                rel_lift_mean=sens.rel_mean,
                hdi_low=sens.rel_hdi_low,
                hdi_high=sens.rel_hdi_high,
            )
        )

    return BayesianResult(
        prior_alpha=config.prior_alpha,
        prior_beta=config.prior_beta,
        control_post_alpha=primary.control_post_alpha,
        control_post_beta=primary.control_post_beta,
        treatment_post_alpha=primary.treatment_post_alpha,
        treatment_post_beta=primary.treatment_post_beta,
        control_post_mean=primary.control_post_mean,
        treatment_post_mean=primary.treatment_post_mean,
        prob_treatment_beats_control=primary.prob_beats,
        expected_loss_treatment=primary.loss_treatment,
        expected_loss_control=primary.loss_control,
        abs_diff_mean=primary.abs_mean,
        abs_diff_hdi_low=primary.abs_hdi_low,
        abs_diff_hdi_high=primary.abs_hdi_high,
        rel_lift_mean=primary.rel_mean,
        rel_lift_hdi_low=primary.rel_hdi_low,
        rel_lift_hdi_high=primary.rel_hdi_high,
        rope_margin=rope_margin,
        prob_in_rope=prob_in_rope,
        prob_above_rope=prob_above_rope,
        prob_below_rope=prob_below_rope,
        sensitivity=sensitivity,
    )
