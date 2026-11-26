"""Statistical power analysis and sample size estimation."""

from __future__ import annotations

import math

from statsmodels.stats.power import NormalIndPower

from src.schemas import ExperimentConfig, PowerAnalysisResult

_CURVE_SAMPLE_SIZES: tuple[int, ...] = (
    1000,
    2500,
    5000,
    7500,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
)
_CURVE_MDES: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20)


def compute_cohens_h(p1: float, p2: float) -> float:
    """Compute signed Cohen's h effect size between two proportions (p2 vs p1)."""
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("Proportions p1 and p2 must lie within [0.0, 1.0].")
    return 2.0 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))


def calculate_sample_size(
    baseline_cr: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute required sample size per variant for a two-sided two-proportion test."""
    if not (0.0 < baseline_cr < 1.0):
        raise ValueError("baseline_cr must be strictly between 0 and 1.")
    if mde_relative <= 0.0:
        raise ValueError("mde_relative must be > 0.")
    target_cr = baseline_cr * (1.0 + mde_relative)
    if target_cr >= 1.0:
        raise ValueError(
            f"Target conversion rate {target_cr:.4f} must be < 1.0."
        )

    effect_size = abs(compute_cohens_h(baseline_cr, target_cr))
    n = NormalIndPower().solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative="two-sided",
    )
    return math.ceil(float(n))


def run_power_analysis(
    config: ExperimentConfig,
    actual_control_n: int,
    actual_treatment_n: int,
) -> PowerAnalysisResult:
    """Evaluate required sample size, achieved power, and power curves."""
    target_cr = config.baseline_cr * (1.0 + config.mde_relative)
    if target_cr >= 1.0:
        raise ValueError(f"Target conversion rate {target_cr:.4f} must be < 1.0.")
    cohens_h = compute_cohens_h(config.baseline_cr, target_cr)
    required_per_variant = calculate_sample_size(
        baseline_cr=config.baseline_cr,
        mde_relative=config.mde_relative,
        alpha=config.alpha,
        power=config.power,
    )
    total_required = 2 * required_per_variant
    estimated_days = max(1, math.ceil(total_required / config.daily_traffic))

    solver = NormalIndPower()
    if actual_control_n > 0 and actual_treatment_n > 0 and abs(cohens_h) > 0.0:
        raw_power = float(
            solver.solve_power(
                effect_size=abs(cohens_h),
                nobs1=actual_control_n,
                alpha=config.alpha,
                ratio=actual_treatment_n / actual_control_n,
                alternative="two-sided",
            )
        )
        achieved_power = min(max(raw_power, 0.0), 1.0)
    else:
        achieved_power = 0.0

    power_curve: list[dict[str, float]] = []
    for mde in _CURVE_MDES:
        curve_target = min(config.baseline_cr * (1.0 + mde), 0.9999)
        curve_h = abs(compute_cohens_h(config.baseline_cr, curve_target))
        for n_per_variant in _CURVE_SAMPLE_SIZES:
            pwr = float(
                solver.solve_power(
                    effect_size=curve_h,
                    nobs1=n_per_variant,
                    alpha=config.alpha,
                    ratio=1.0,
                    alternative="two-sided",
                )
            )
            power_curve.append(
                {
                    "n_per_variant": float(n_per_variant),
                    "mde_relative": float(mde),
                    "power": min(max(pwr, 0.0), 1.0),
                }
            )

    return PowerAnalysisResult(
        baseline_cr=config.baseline_cr,
        target_cr=target_cr,
        mde_relative=config.mde_relative,
        cohens_h=cohens_h,
        alpha=config.alpha,
        target_power=config.power,
        required_sample_per_variant=required_per_variant,
        total_required_sample=total_required,
        estimated_duration_days=estimated_days,
        actual_sample_per_variant=min(actual_control_n, actual_treatment_n),
        achieved_power=achieved_power,
        power_curve=power_curve,
    )
