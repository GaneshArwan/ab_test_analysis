"""Unit and statistical-reference tests for power, frequentist, and Bayesian modules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from src.bayesian import compute_hdi, run_bayesian_analysis
from src.frequentist import (
    check_srm,
    newcombe_wilson_diff_ci,
    run_frequentist_analysis,
    wilson_ci,
)
from src.power import (
    _CURVE_MDES,
    _CURVE_SAMPLE_SIZES,
    calculate_sample_size,
    compute_cohens_h,
    run_power_analysis,
)
from src.schemas import ExperimentConfig


@pytest.fixture
def synthetic_controlled_df() -> pd.DataFrame:
    """Build a deterministic controlled DataFrame across devices and channels."""
    rows: list[dict[str, object]] = []
    devices = ["desktop", "mobile", "tablet"]
    channels = ["organic", "paid_search", "email", "social"]

    idx = 0
    for arm, n_arm, n_conv in [("control", 600, 60), ("treatment", 600, 96)]:
        for i in range(n_arm):
            converted = i < n_conv
            device = devices[i % len(devices)]
            channel = channels[i % len(channels)]
            revenue = float(40.0 + (i % 7) * 5.0) if converted else 0.0
            rows.append(
                {
                    "user_id": f"u_{idx:05d}",
                    "timestamp": pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=idx),
                    "group": arm,
                    "device": device,
                    "channel": channel,
                    "viewed_landing": True,
                    "clicked_cta": True,
                    "started_checkout": True,
                    "converted": converted,
                    "revenue": revenue,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_compute_cohens_h_zero_and_antisymmetric() -> None:
    # Arrange
    p1, p2 = 0.10, 0.15

    # Act
    h_same = compute_cohens_h(p1, p1)
    h_fwd = compute_cohens_h(p1, p2)
    h_rev = compute_cohens_h(p2, p1)

    # Assert
    assert h_same == pytest.approx(0.0)
    assert h_fwd > 0.0
    assert h_fwd == pytest.approx(-h_rev, rel=1e-12)


@pytest.mark.parametrize(("p1", "p2"), [(-0.01, 0.1), (0.1, 1.01)])
def test_compute_cohens_h_invalid_proportions_raise_value_error(
    p1: float, p2: float
) -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Proportions p1 and p2"):
        compute_cohens_h(p1, p2)


def test_calculate_sample_size_matches_normal_ind_power_and_monotonicity() -> None:
    # Arrange
    baseline_cr = 0.10
    mde_small = 0.10
    mde_large = 0.20
    alpha = 0.05
    target_cr_small = baseline_cr * (1.0 + mde_small)
    expected_h = abs(compute_cohens_h(baseline_cr, target_cr_small))
    ref_n = math.ceil(
        float(
            NormalIndPower().solve_power(
                effect_size=expected_h,
                power=0.80,
                alpha=alpha,
                ratio=1.0,
                alternative="two-sided",
            )
        )
    )

    # Act
    n_small_mde_80 = calculate_sample_size(baseline_cr, mde_small, alpha=alpha, power=0.80)
    n_large_mde_80 = calculate_sample_size(baseline_cr, mde_large, alpha=alpha, power=0.80)
    n_small_mde_90 = calculate_sample_size(baseline_cr, mde_small, alpha=alpha, power=0.90)

    # Assert
    assert n_small_mde_80 == ref_n
    assert n_small_mde_80 > n_large_mde_80
    assert n_small_mde_90 > n_small_mde_80


@pytest.mark.parametrize(
    ("baseline_cr", "mde_relative"),
    [(0.0, 0.1), (1.0, 0.1), (0.1, 0.0), (0.1, -0.2), (0.8, 0.5)],
)
def test_calculate_sample_size_invalid_args_raise_value_error(
    baseline_cr: float, mde_relative: float
) -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        calculate_sample_size(baseline_cr=baseline_cr, mde_relative=mde_relative)


def test_run_power_analysis_curve_length_bounds_and_zero_arm() -> None:
    # Arrange
    cfg = ExperimentConfig(n_users=10000, baseline_cr=0.10, mde_relative=0.12)

    # Act
    res = run_power_analysis(cfg, actual_control_n=5000, actual_treatment_n=5000)
    res_zero = run_power_analysis(cfg, actual_control_n=0, actual_treatment_n=5000)

    # Assert
    assert len(res.power_curve) == len(_CURVE_MDES) * len(_CURVE_SAMPLE_SIZES)
    assert all(0.0 <= pt["power"] <= 1.0 for pt in res.power_curve)
    assert 0.0 < res.achieved_power <= 1.0
    assert res.total_required_sample == 2 * res.required_sample_per_variant
    assert res_zero.achieved_power == pytest.approx(0.0)


def test_check_srm_balanced_passes_and_imbalanced_fails() -> None:
    # Arrange
    balanced_c, balanced_t = 5000, 5000
    imbalanced_c, imbalanced_t = 5500, 4500

    # Act
    pass_res = check_srm(balanced_c, balanced_t)
    fail_res = check_srm(imbalanced_c, imbalanced_t)

    # Assert
    assert pass_res.passed is True
    assert pass_res.chi2_stat == pytest.approx(0.0)
    assert pass_res.p_value == pytest.approx(1.0)
    assert fail_res.passed is False
    assert fail_res.p_value < 0.001


@pytest.mark.parametrize(
    ("ctrl_n", "trt_n", "ratio", "alpha"),
    [(0, 0, 0.5, 0.001), (-1, 10, 0.5, 0.001), (10, 10, 1.0, 0.001), (10, 10, 0.5, 0.0)],
)
def test_check_srm_invalid_inputs_raise_value_error(
    ctrl_n: int, trt_n: int, ratio: float, alpha: float
) -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        check_srm(ctrl_n, trt_n, expected_ratio=ratio, alpha=alpha)


@pytest.mark.parametrize(
    ("successes", "total"),
    [(0, 100), (100, 100), (15, 100)],
)
def test_wilson_ci_boundaries_and_containment(successes: int, total: int) -> None:
    # Arrange
    p = successes / total

    # Act
    low, high = wilson_ci(successes, total, alpha=0.05)

    # Assert
    assert 0.0 <= low <= high <= 1.0
    assert low <= p + 1e-12
    assert high >= p - 1e-12
    if successes == 0:
        assert low == pytest.approx(0.0, abs=1e-12)
    if successes == total:
        assert high == pytest.approx(1.0, abs=1e-12)


def test_wilson_ci_invalid_args_raise_value_error() -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2"),
    [(0, 100, 0, 100), (100, 100, 100, 100), (10, 100, 20, 100)],
)
def test_newcombe_wilson_diff_ci_bounds_and_containment(
    k1: int, n1: int, k2: int, n2: int
) -> None:
    # Arrange
    diff = (k2 / n2) - (k1 / n1)

    # Act
    low, high = newcombe_wilson_diff_ci(k1, n1, k2, n2, alpha=0.05)

    # Assert
    assert -1.0 <= low <= diff <= high <= 1.0


def test_run_frequentist_analysis_matches_reference_tests_and_bh_fdr(
    synthetic_controlled_df: pd.DataFrame,
) -> None:
    # Arrange
    ctrl_rev = synthetic_controlled_df.loc[
        synthetic_controlled_df["group"] == "control", "revenue"
    ].to_numpy(dtype=float)
    trt_rev = synthetic_controlled_df.loc[
        synthetic_controlled_df["group"] == "treatment", "revenue"
    ].to_numpy(dtype=float)
    ref_z, ref_zp = proportions_ztest([96, 60], [600, 600], alternative="two-sided")
    ref_welch = stats.ttest_ind(trt_rev, ctrl_rev, equal_var=False)

    # Act
    res = run_frequentist_analysis(synthetic_controlled_df, alpha=0.05)

    # Assert
    assert res.control_n == 600
    assert res.treatment_n == 600
    assert res.control_cr == pytest.approx(0.10)
    assert res.treatment_cr == pytest.approx(0.16)
    assert res.z_stat == pytest.approx(float(ref_z), rel=1e-9)
    assert res.z_p_value == pytest.approx(float(ref_zp), rel=1e-9)
    assert res.rpu_welch_t == pytest.approx(float(ref_welch.statistic), rel=1e-9)
    assert res.rpu_welch_p == pytest.approx(float(ref_welch.pvalue), rel=1e-9)
    assert res.significant is True
    assert len(res.funnel_stages) == 4
    assert len(res.segments) == 7
    assert all(seg.p_value_bh >= seg.p_value - 1e-12 for seg in res.segments)


def test_run_frequentist_analysis_zero_conversion_and_empty_arm_edge_cases(
    synthetic_controlled_df: pd.DataFrame,
) -> None:
    # Arrange
    zero_df = synthetic_controlled_df.copy()
    zero_df["converted"] = False
    zero_df["revenue"] = 0.0
    only_ctrl_df = zero_df.loc[zero_df["group"] == "control"].copy()

    # Act
    zero_res = run_frequentist_analysis(zero_df, alpha=0.05)

    # Assert
    assert zero_res.z_stat == pytest.approx(0.0)
    assert zero_res.z_p_value == pytest.approx(1.0)
    assert zero_res.chi2_p_value == pytest.approx(1.0)
    assert zero_res.rpu_welch_p == pytest.approx(1.0)
    assert zero_res.rpu_mannwhitney_p == pytest.approx(1.0)
    assert zero_res.control_aov == pytest.approx(0.0)
    assert zero_res.treatment_aov == pytest.approx(0.0)

    with pytest.raises(ValueError, match="non-empty"):
        run_frequentist_analysis(only_ctrl_df)


def test_compute_hdi_produces_shortest_interval_on_skewed_distribution() -> None:
    # Arrange
    rng = np.random.default_rng(7)
    samples = rng.exponential(scale=1.0, size=20000)
    et_low, et_high = np.quantile(samples, [0.025, 0.975])

    # Act
    hdi_low, hdi_high = compute_hdi(samples, credible_mass=0.95)

    # Assert
    assert hdi_low < hdi_high
    assert (hdi_high - hdi_low) <= (et_high - et_low) + 1e-6
    assert hdi_low < et_low  # right-skewed mode near 0 pulls HDI left


@pytest.mark.parametrize(
    ("samples", "mass"),
    [(np.array([1.0]), 0.95), (np.array([1.0, 2.0, 3.0]), 1.0), (np.array([1.0, 2.0, 3.0]), 0.0)],
)
def test_compute_hdi_invalid_inputs_raise_value_error(
    samples: np.ndarray, mass: float
) -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        compute_hdi(samples, credible_mass=mass)


def test_run_bayesian_analysis_deterministic_rope_sum_and_sensitivity() -> None:
    # Arrange
    cfg = ExperimentConfig(seed=99, rope_margin=0.005)

    # Act
    res_1 = run_bayesian_analysis(
        control_n=5000,
        control_conv=500,
        treatment_n=5000,
        treatment_conv=650,
        config=cfg,
        n_draws=20000,
    )
    res_2 = run_bayesian_analysis(
        control_n=5000,
        control_conv=500,
        treatment_n=5000,
        treatment_conv=650,
        config=cfg,
        n_draws=20000,
    )

    # Assert
    assert res_1 == res_2
    assert res_1.prob_treatment_beats_control > 0.99
    assert res_1.expected_loss_treatment < res_1.expected_loss_control
    rope_sum = res_1.prob_in_rope + res_1.prob_above_rope + res_1.prob_below_rope
    assert rope_sum == pytest.approx(1.0, abs=1e-6)
    assert len(res_1.sensitivity) == 3


@pytest.mark.parametrize(
    ("ctrl_n", "ctrl_k", "trt_n", "trt_k", "n_draws"),
    [(0, 0, 100, 10, 1000), (100, 101, 100, 10, 1000), (100, 10, 100, 10, 50)],
)
def test_run_bayesian_analysis_invalid_inputs_raise_value_error(
    ctrl_n: int, ctrl_k: int, trt_n: int, trt_k: int, n_draws: int
) -> None:
    # Arrange
    cfg = ExperimentConfig()

    # Act & Assert
    with pytest.raises(ValueError):
        run_bayesian_analysis(ctrl_n, ctrl_k, trt_n, trt_k, config=cfg, n_draws=n_draws)
