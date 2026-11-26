"""Frequentist hypothesis testing, confidence intervals, funnel, and FDR segment analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from src.power import compute_cohens_h
from src.schemas import (
    FrequentistResult,
    FunnelStageResult,
    SegmentResult,
    SRMResult,
)

_FUNNEL_STAGES: tuple[str, ...] = (
    "viewed_landing",
    "clicked_cta",
    "started_checkout",
    "converted",
)
_SEGMENT_DIMENSIONS: tuple[str, ...] = ("device", "channel")


def check_srm(
    control_n: int,
    treatment_n: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.001,
) -> SRMResult:
    """Run chi-square goodness-of-fit test for Sample Ratio Mismatch."""
    if control_n < 0 or treatment_n < 0 or (control_n + treatment_n) == 0:
        raise ValueError("Arm counts must be non-negative with total > 0.")
    if not (0.0 < expected_ratio < 1.0):
        raise ValueError("expected_ratio must lie strictly in (0, 1).")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly in (0, 1).")

    total = control_n + treatment_n
    expected = [total * expected_ratio, total * (1.0 - expected_ratio)]
    res = stats.chisquare([control_n, treatment_n], f_exp=expected)
    chi2_stat = float(res.statistic)
    p_value = float(res.pvalue)
    return SRMResult(
        control_n=control_n,
        treatment_n=treatment_n,
        expected_ratio=expected_ratio,
        chi2_stat=chi2_stat,
        p_value=p_value,
        alpha=alpha,
        passed=bool(p_value >= alpha),
    )


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be > 0.")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total].")

    p = successes / total
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half_width = (
        z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    ) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def newcombe_wilson_diff_ci(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute Newcombe-Wilson score CI for difference of proportions (p2 - p1)."""
    l1, u1 = wilson_ci(k1, n1, alpha=alpha)
    l2, u2 = wilson_ci(k2, n2, alpha=alpha)
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p2 - p1
    low = diff - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    high = diff + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return max(-1.0, low), min(1.0, high)


def _safe_two_prop_ztest(
    trt_k: int, trt_n: int, ctrl_k: int, ctrl_n: int
) -> tuple[float, float]:
    if (trt_k == 0 and ctrl_k == 0) or (trt_k == trt_n and ctrl_k == ctrl_n):
        return 0.0, 1.0
    z_stat, p_val = proportions_ztest(
        [trt_k, ctrl_k], [trt_n, ctrl_n], alternative="two-sided"
    )
    z_f = float(z_stat)
    p_f = float(p_val)
    if not (math.isfinite(z_f) and math.isfinite(p_f)):
        return 0.0, 1.0
    return z_f, p_f


def _safe_chi2_test(
    ctrl_k: int, ctrl_n: int, trt_k: int, trt_n: int
) -> tuple[float, float]:
    if (ctrl_k + trt_k) == 0 or (ctrl_k + trt_k) == (ctrl_n + trt_n):
        return 0.0, 1.0
    table = [[ctrl_k, ctrl_n - ctrl_k], [trt_k, trt_n - trt_k]]
    chi2_stat, p_val, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2_stat), float(p_val)


def run_frequentist_analysis(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Run frequentist proportion, revenue, funnel, and BH-adjusted segment tests."""
    ctrl = df.loc[df["group"] == "control"]
    trt = df.loc[df["group"] == "treatment"]

    ctrl_n = len(ctrl)
    trt_n = len(trt)
    if ctrl_n == 0 or trt_n == 0:
        raise ValueError("Both control and treatment groups must be non-empty.")

    ctrl_conv = int(ctrl["converted"].sum())
    trt_conv = int(trt["converted"].sum())
    ctrl_cr = ctrl_conv / ctrl_n
    trt_cr = trt_conv / trt_n

    ctrl_ci_low, ctrl_ci_high = wilson_ci(ctrl_conv, ctrl_n, alpha=alpha)
    trt_ci_low, trt_ci_high = wilson_ci(trt_conv, trt_n, alpha=alpha)
    abs_diff = trt_cr - ctrl_cr
    rel_lift = (abs_diff / ctrl_cr) if ctrl_cr > 0.0 else 0.0
    diff_ci_low, diff_ci_high = newcombe_wilson_diff_ci(
        ctrl_conv, ctrl_n, trt_conv, trt_n, alpha=alpha
    )

    z_stat, z_p_value = _safe_two_prop_ztest(trt_conv, trt_n, ctrl_conv, ctrl_n)
    chi2_stat, chi2_p_value = _safe_chi2_test(ctrl_conv, ctrl_n, trt_conv, trt_n)
    cohens_h = compute_cohens_h(ctrl_cr, trt_cr)
    significant = bool(z_p_value < alpha)

    # Revenue metrics (RPU and AOV).
    ctrl_rev = ctrl["revenue"].to_numpy(dtype=float)
    trt_rev = trt["revenue"].to_numpy(dtype=float)
    ctrl_rpu = float(np.mean(ctrl_rev))
    trt_rpu = float(np.mean(trt_rev))
    rpu_diff = trt_rpu - ctrl_rpu

    var_c = float(np.var(ctrl_rev, ddof=1)) if ctrl_n > 1 else 0.0
    var_t = float(np.var(trt_rev, ddof=1)) if trt_n > 1 else 0.0
    se_c = var_c / ctrl_n
    se_t = var_t / trt_n
    se_diff = math.sqrt(se_c + se_t)

    if se_diff > 0.0 and ctrl_n > 1 and trt_n > 1:
        denom = (se_c**2) / (ctrl_n - 1) + (se_t**2) / (trt_n - 1)
        welch_df = ((se_c + se_t) ** 2) / denom if denom > 0.0 else (ctrl_n + trt_n - 2)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=welch_df))
        rpu_ci_low = rpu_diff - t_crit * se_diff
        rpu_ci_high = rpu_diff + t_crit * se_diff
        welch_res = stats.ttest_ind(trt_rev, ctrl_rev, equal_var=False)
        welch_t = float(welch_res.statistic)
        welch_p = float(welch_res.pvalue)
    else:
        rpu_ci_low = rpu_diff
        rpu_ci_high = rpu_diff
        welch_t = 0.0
        welch_p = 1.0

    if np.all(ctrl_rev == trt_rev[0]) and np.all(trt_rev == trt_rev[0]):
        mw_u, mw_p = 0.0, 1.0
    else:
        mw_res = stats.mannwhitneyu(trt_rev, ctrl_rev, alternative="two-sided")
        mw_u = float(mw_res.statistic)
        mw_p = float(mw_res.pvalue)

    if ctrl_n + trt_n > 2:
        pooled_var = ((ctrl_n - 1) * var_c + (trt_n - 1) * var_t) / (
            ctrl_n + trt_n - 2
        )
        pooled_sd = math.sqrt(pooled_var)
        rpu_cohens_d = (rpu_diff / pooled_sd) if pooled_sd > 0.0 else 0.0
    else:
        rpu_cohens_d = 0.0

    ctrl_conv_rev = ctrl_rev[ctrl["converted"].to_numpy(dtype=bool)]
    trt_conv_rev = trt_rev[trt["converted"].to_numpy(dtype=bool)]
    ctrl_aov = float(np.mean(ctrl_conv_rev)) if len(ctrl_conv_rev) > 0 else 0.0
    trt_aov = float(np.mean(trt_conv_rev)) if len(trt_conv_rev) > 0 else 0.0

    # Funnel stages.
    funnel_stages: list[FunnelStageResult] = []
    for stage in _FUNNEL_STAGES:
        c_stage_n = int(ctrl[stage].sum())
        t_stage_n = int(trt[stage].sum())
        c_rate = c_stage_n / ctrl_n
        t_rate = t_stage_n / trt_n
        stage_lift = ((t_rate - c_rate) / c_rate) if c_rate > 0.0 else 0.0
        _, stage_p = _safe_two_prop_ztest(t_stage_n, trt_n, c_stage_n, ctrl_n)
        funnel_stages.append(
            FunnelStageResult(
                stage=stage,
                control_n=c_stage_n,
                control_rate=c_rate,
                treatment_n=t_stage_n,
                treatment_rate=t_rate,
                relative_lift=stage_lift,
                p_value=stage_p,
            )
        )

    # Segments with Benjamini-Hochberg FDR adjustment across all subgroups.
    raw_segments: list[SegmentResult] = []
    raw_pvals: list[float] = []
    for dim in _SEGMENT_DIMENSIONS:
        for seg_val in sorted(df[dim].dropna().unique()):
            seg_c = ctrl.loc[ctrl[dim] == seg_val]
            seg_t = trt.loc[trt[dim] == seg_val]
            sc_n = len(seg_c)
            st_n = len(seg_t)
            if sc_n == 0 or st_n == 0:
                continue
            sc_k = int(seg_c["converted"].sum())
            st_k = int(seg_t["converted"].sum())
            sc_cr = sc_k / sc_n
            st_cr = st_k / st_n
            s_diff = st_cr - sc_cr
            s_lift = (s_diff / sc_cr) if sc_cr > 0.0 else 0.0
            s_ci_low, s_ci_high = newcombe_wilson_diff_ci(
                sc_k, sc_n, st_k, st_n, alpha=alpha
            )
            _, s_p = _safe_two_prop_ztest(st_k, st_n, sc_k, sc_n)
            raw_pvals.append(s_p)
            raw_segments.append(
                SegmentResult(
                    dimension=dim,
                    segment=str(seg_val),
                    control_n=sc_n,
                    control_cr=sc_cr,
                    treatment_n=st_n,
                    treatment_cr=st_cr,
                    abs_diff=s_diff,
                    rel_lift=s_lift,
                    ci_low=s_ci_low,
                    ci_high=s_ci_high,
                    p_value=s_p,
                    p_value_bh=s_p,
                    significant_bh=False,
                )
            )

    segments: list[SegmentResult] = []
    if raw_pvals:
        rejected, pvals_bh, _, _ = multipletests(
            raw_pvals, alpha=alpha, method="fdr_bh"
        )
        segments = [
            seg.model_copy(
                update={
                    "p_value_bh": float(p_bh),
                    "significant_bh": bool(sig_bh),
                }
            )
            for seg, p_bh, sig_bh in zip(raw_segments, pvals_bh, rejected)
        ]

    return FrequentistResult(
        control_n=ctrl_n,
        control_conversions=ctrl_conv,
        control_cr=ctrl_cr,
        control_ci_low=ctrl_ci_low,
        control_ci_high=ctrl_ci_high,
        treatment_n=trt_n,
        treatment_conversions=trt_conv,
        treatment_cr=trt_cr,
        treatment_ci_low=trt_ci_low,
        treatment_ci_high=trt_ci_high,
        abs_diff=abs_diff,
        rel_lift=rel_lift,
        diff_ci_low=diff_ci_low,
        diff_ci_high=diff_ci_high,
        z_stat=z_stat,
        z_p_value=z_p_value,
        chi2_stat=chi2_stat,
        chi2_p_value=chi2_p_value,
        cohens_h=cohens_h,
        significant=significant,
        control_rpu=ctrl_rpu,
        treatment_rpu=trt_rpu,
        rpu_diff=rpu_diff,
        rpu_ci_low=rpu_ci_low,
        rpu_ci_high=rpu_ci_high,
        rpu_welch_t=welch_t,
        rpu_welch_p=welch_p,
        rpu_mannwhitney_u=mw_u,
        rpu_mannwhitney_p=mw_p,
        rpu_cohens_d=rpu_cohens_d,
        control_aov=ctrl_aov,
        treatment_aov=trt_aov,
        funnel_stages=funnel_stages,
        segments=segments,
    )
