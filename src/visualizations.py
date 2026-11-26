"""Publication-quality static charts for A/B experiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.i18n import fmt_pct, t
from src.schemas import BayesianResult, FrequentistResult, PowerAnalysisResult

_CTRL_COLOR = "#0072B2"
_TRT_COLOR = "#D55E00"
_TEAL_COLOR = "#009E73"
_SLATE_COLOR = "#475569"
_MUTED_COLOR = "#94A3B8"
_ROPE_COLOR = "#E69F00"
_CURVE_PALETTE: tuple[str, ...] = ("#64748B", "#0072B2", "#009E73", "#D55E00")

_STAGE_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "viewed_landing": "Viewed Landing",
        "clicked_cta": "Clicked CTA",
        "started_checkout": "Started Checkout",
        "converted": "Converted",
    },
    "id": {
        "viewed_landing": "Lihat Halaman",
        "clicked_cta": "Klik CTA",
        "started_checkout": "Mulai Checkout",
        "converted": "Konversi",
    },
}


def _apply_clean_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#1E293B", labelsize=9.5)
    ax.grid(axis="both", linestyle="--", alpha=0.35, color="#CBD5E1")
    ax.set_axisbelow(True)


def plot_conversion_funnel(
    frequentist: FrequentistResult,
    output_path: Path,
    locale: str = "en",
) -> Path:
    """Plot Control vs Treatment funnel step rates with relative lift annotations."""
    stages = frequentist.funnel_stages
    labels = [
        _STAGE_LABELS.get(locale, _STAGE_LABELS["en"]).get(s.stage, s.stage)
        for s in stages
    ]
    ctrl_rates = [s.control_rate * 100.0 for s in stages]
    trt_rates = [s.treatment_rate * 100.0 for s in stages]

    x = np.arange(len(stages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    _apply_clean_style(ax)

    bars_c = ax.bar(
        x - width / 2,
        ctrl_rates,
        width,
        label=t("control", locale),
        color=_CTRL_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    bars_t = ax.bar(
        x + width / 2,
        trt_rates,
        width,
        label=t("treatment", locale),
        color=_TRT_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )

    for idx, (bc, bt, stage_res) in enumerate(zip(bars_c, bars_t, stages)):
        ax.text(
            bc.get_x() + bc.get_width() / 2,
            bc.get_height() + 1.2,
            f"{ctrl_rates[idx]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1E293B",
        )
        ax.text(
            bt.get_x() + bt.get_width() / 2,
            bt.get_height() + 1.2,
            f"{trt_rates[idx]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1E293B",
            fontweight="bold",
        )
        sig_marker = "*" if stage_res.p_value < 0.05 else ""
        lift_pct = stage_res.relative_lift * 100.0
        top_y = max(ctrl_rates[idx], trt_rates[idx]) + 6.5
        badge_color = _TEAL_COLOR if lift_pct > 0 and stage_res.p_value < 0.05 else _SLATE_COLOR
        ax.text(
            x[idx],
            top_y,
            f"{lift_pct:+.1f}%{sig_marker}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=badge_color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#F8FAFC",
                "edgecolor": badge_color,
                "linewidth": 0.8,
            },
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_ylabel(f"{t('rate', locale)} (%)", fontsize=10, fontweight="bold", color="#0F172A")
    title = (
        "Conversion Funnel: Control vs. Treatment"
        if locale == "en"
        else "Funnel Konversi: Kontrol vs. Perlakuan"
    )
    subtitle = (
        "Step reach rates (%) and relative treatment lift (* p < 0.05)"
        if locale == "en"
        else "Tingkat pencapaian tahap (%) dan peningkatan relatif (* p < 0,05)"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_frequentist_segments(
    frequentist: FrequentistResult,
    output_path: Path,
    locale: str = "en",
) -> Path:
    """Plot forest chart of Overall + Segment absolute CR lifts (pp) with 95% CIs."""
    overall_label = t("overall_label", locale)
    rows: list[tuple[str, float, float, float, bool]] = [
        (
            overall_label,
            frequentist.abs_diff * 100.0,
            frequentist.diff_ci_low * 100.0,
            frequentist.diff_ci_high * 100.0,
            frequentist.significant,
        )
    ]
    for seg in frequentist.segments:
        label = f"{seg.dimension.title()}: {seg.segment.replace('_', ' ').title()}"
        rows.append(
            (
                label,
                seg.abs_diff * 100.0,
                seg.ci_low * 100.0,
                seg.ci_high * 100.0,
                seg.significant_bh,
            )
        )

    rows_rev = list(reversed(rows))
    y_pos = np.arange(len(rows_rev))

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    _apply_clean_style(ax)

    ax.axvline(0.0, color="#64748B", linestyle="--", linewidth=1.2, zorder=1)

    all_lows = [r[2] for r in rows_rev]
    all_highs = [r[3] for r in rows_rev]
    x_min = min(all_lows + [-0.5]) - 0.5
    x_max = max(all_highs + [1.0]) + 2.4

    for idx, (label, diff, low, high, is_sig) in enumerate(rows_rev):
        is_overall = label == overall_label
        color = _TRT_COLOR if is_overall else (_TEAL_COLOR if is_sig else _MUTED_COLOR)
        marker = "D" if is_overall else "o"
        msize = 8.0 if is_overall else 6.8
        lw = 2.4 if is_overall else 1.8

        ax.errorbar(
            diff,
            idx,
            xerr=[[max(0.0, diff - low)], [max(0.0, high - diff)]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=lw,
            capsize=4,
            markersize=msize,
            zorder=3,
        )
        sig_tag = (
            (t("sig_abbr", locale) if is_overall else t("bh_sig_abbr", locale))
            if is_sig
            else t("ns_abbr", locale)
        )
        ax.text(
            high + 0.18,
            idx,
            f"{diff:+.2f} pp [{low:+.2f}, {high:+.2f}] ({sig_tag})",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#0F172A" if is_sig else "#475569",
            fontweight="bold" if is_overall or is_sig else "normal",
        )

    if len(rows_rev) > 1:
        ax.axhline(len(rows_rev) - 1.5, color="#E2E8F0", linestyle="-", linewidth=1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows_rev], fontsize=9.5)
    ax.set_xlim(x_min, x_max)
    xlabel = (
        "Absolute Conversion Lift (Percentage Points, 95% Newcombe-Wilson CI)"
        if locale == "en"
        else "Selisih Konversi Absolut (Poin Persentase, CI 95% Newcombe-Wilson)"
    )
    title = (
        "Conversion Lift Across Overall Population & Subgroups"
        if locale == "en"
        else "Peningkatan Konversi pada Populasi Total & Subkelompok"
    )
    subtitle = (
        "Subgroup significance controlled via Benjamini-Hochberg FDR (alpha = 0.05)"
        if locale == "en"
        else "Signifikansi subkelompok dikontrol dengan Benjamini-Hochberg FDR (alfa = 0,05)"
    )
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_bayesian_posteriors(
    bayesian: BayesianResult,
    output_path: Path,
    locale: str = "en",
    seed: int = 42,
) -> Path:
    """Plot 2-panel Bayesian Beta posteriors and relative lift distribution with 95% HDI & ROPE."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    _apply_clean_style(ax_left)
    _apply_clean_style(ax_right)

    # Panel (a): Beta posterior densities for Control and Treatment CR (%).
    c_dist = stats.beta(bayesian.control_post_alpha, bayesian.control_post_beta)
    t_dist = stats.beta(bayesian.treatment_post_alpha, bayesian.treatment_post_beta)

    c_low, c_high = c_dist.ppf([0.0005, 0.9995])
    t_low, t_high = t_dist.ppf([0.0005, 0.9995])
    x_min = max(0.0, min(c_low, t_low) - 0.002)
    x_max = min(1.0, max(c_high, t_high) + 0.002)
    x_grid = np.linspace(x_min, x_max, 500)
    x_pct = x_grid * 100.0

    y_c = c_dist.pdf(x_grid) / 100.0
    y_t = t_dist.pdf(x_grid) / 100.0

    ax_left.plot(x_pct, y_c, color=_CTRL_COLOR, linewidth=2.2, label=t("control", locale))
    ax_left.fill_between(x_pct, 0.0, y_c, color=_CTRL_COLOR, alpha=0.22)
    ax_left.plot(x_pct, y_t, color=_TRT_COLOR, linewidth=2.2, label=t("treatment", locale))
    ax_left.fill_between(x_pct, 0.0, y_t, color=_TRT_COLOR, alpha=0.22)

    ax_left.axvline(bayesian.control_post_mean * 100.0, color=_CTRL_COLOR, linestyle="--", linewidth=1.2)
    ax_left.axvline(bayesian.treatment_post_mean * 100.0, color=_TRT_COLOR, linestyle="--", linewidth=1.2)
    max_y_left = float(max(np.max(y_c), np.max(y_t)))
    ax_left.set_ylim(0.0, max_y_left * 1.26)

    ax_left.set_xlabel(f"{t('conversion_rate', locale)} (%)", fontsize=9.5, fontweight="bold")
    ax_left.set_ylabel("Posterior Density" if locale == "en" else "Densitas Posterior", fontsize=9.5, fontweight="bold")
    ax_left.set_title(
        "(a) Posterior Conversion Rate Distributions"
        if locale == "en"
        else "(a) Distribusi Posterior Tingkat Konversi",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
    )
    ax_left.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right")

    info_text = (
        f"{t('prob_to_beat', locale)}: {fmt_pct(bayesian.prob_treatment_beats_control, locale, 2)}\n"
        f"{t('expected_loss', locale)}: {fmt_pct(bayesian.expected_loss_treatment, locale, 4)}"
    )
    ax_left.text(
        0.03,
        0.95,
        info_text,
        transform=ax_left.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    # Panel (b): Posterior relative lift (%) distribution with 95% HDI & ROPE.
    rng = np.random.default_rng(seed)
    p_c = rng.beta(bayesian.control_post_alpha, bayesian.control_post_beta, size=100000)
    p_t = rng.beta(bayesian.treatment_post_alpha, bayesian.treatment_post_beta, size=100000)
    rel_lift_pct = np.where(p_c > 0.0, (p_t - p_c) / p_c * 100.0, 0.0)

    kde = stats.gaussian_kde(rel_lift_pct)
    r_min = float(np.percentile(rel_lift_pct, 0.1))
    r_max = float(np.percentile(rel_lift_pct, 99.9))
    rope_rel = (
        (bayesian.rope_margin / bayesian.control_post_mean) * 100.0
        if bayesian.control_post_mean > 0.0
        else 0.0
    )
    grid_lo = min(r_min, -rope_rel - 1.5)
    grid_hi = max(r_max, rope_rel + 1.5)
    r_grid = np.linspace(grid_lo, grid_hi, 500)
    r_pdf = kde(r_grid)

    hdi_lo = bayesian.rel_lift_hdi_low * 100.0
    hdi_hi = bayesian.rel_lift_hdi_high * 100.0
    mean_rel = bayesian.rel_lift_mean * 100.0

    ax_right.plot(r_grid, r_pdf, color="#0F172A", linewidth=2.0)
    hdi_mask = (r_grid >= hdi_lo) & (r_grid <= hdi_hi)
    ax_right.fill_between(
        r_grid[hdi_mask],
        0.0,
        r_pdf[hdi_mask],
        color=_TEAL_COLOR,
        alpha=0.28,
        label=f"{t('hdi_95', locale)}: [{hdi_lo:+.1f}%, {hdi_hi:+.1f}%]",
    )
    ax_right.axvspan(
        -rope_rel,
        rope_rel,
        color=_ROPE_COLOR,
        alpha=0.18,
        label=f"{t('rope', locale)} (±{rope_rel:.1f}%)",
    )
    ax_right.axvline(0.0, color="#64748B", linestyle=":", linewidth=1.1)
    ax_right.axvline(
        mean_rel,
        color=_TEAL_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"{t('mean_label', locale)}: {mean_rel:+.1f}%",
    )
    ax_right.set_ylim(0.0, float(np.max(r_pdf)) * 1.28)

    ax_right.set_xlabel(f"{t('relative_lift', locale)} (%)", fontsize=9.5, fontweight="bold")
    ax_right.set_ylabel("Density" if locale == "en" else "Densitas", fontsize=9.5, fontweight="bold")
    ax_right.set_title(
        "(b) Posterior Relative Lift with 95% HDI & ROPE"
        if locale == "en"
        else "(b) Posterior Peningkatan Relatif (HDI 95% & ROPE)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
    )
    ax_right.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", fontsize=8.0)

    rope_box = (
        f"P(> ROPE): {fmt_pct(bayesian.prob_above_rope, locale, 1)}\n"
        f"P(in ROPE): {fmt_pct(bayesian.prob_in_rope, locale, 1)}"
    )
    ax_right.text(
        0.03,
        0.95,
        rope_box,
        transform=ax_right.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_power_curve(
    power: PowerAnalysisResult,
    output_path: Path,
    locale: str = "en",
) -> Path:
    """Plot statistical power vs. sample size per variant across MDE targets."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    _apply_clean_style(ax)

    mdes = sorted({row["mde_relative"] for row in power.power_curve})
    for idx, mde in enumerate(mdes):
        pts = sorted(
            (row for row in power.power_curve if row["mde_relative"] == mde),
            key=lambda r: r["n_per_variant"],
        )
        ns = [p["n_per_variant"] for p in pts]
        pwrs = [p["power"] * 100.0 for p in pts]
        color = _CURVE_PALETTE[idx % len(_CURVE_PALETTE)]
        is_target = abs(mde - power.mde_relative) < 1e-6
        ax.plot(
            ns,
            pwrs,
            marker="o",
            markersize=4.5,
            linewidth=2.6 if is_target else 1.8,
            color=color,
            label=f"MDE {mde * 100:.0f}%{' (Target)' if is_target else ''}",
        )

    target_pwr_pct = power.target_power * 100.0
    ax.axhline(
        target_pwr_pct,
        color="#DC2626",
        linestyle="--",
        linewidth=1.3,
        label=f"{target_pwr_pct:.0f}% {t('target_power_label', locale)}",
    )

    if power.actual_sample_per_variant > 0:
        ax.axvline(
            power.actual_sample_per_variant,
            color="#0F172A",
            linestyle="-.",
            linewidth=1.5,
            label=(
                f"Observed N/Variant = {power.actual_sample_per_variant:,}"
                if locale == "en"
                else f"Sampel Aktual/Varian = {power.actual_sample_per_variant:,}"
            ),
        )

    ax.set_ylim(0, 105)
    ax.set_xlabel(
        "Sample Size per Variant" if locale == "en" else "Ukuran Sampel per Varian",
        fontsize=9.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax.set_ylabel(
        "Statistical Power (%)" if locale == "en" else "Daya Statistik (%)",
        fontsize=9.5,
        fontweight="bold",
        color="#0F172A",
    )
    title = (
        "Statistical Power Curve by Minimum Detectable Effect (MDE)"
        if locale == "en"
        else "Kurva Daya Statistik Berdasarkan Minimum Detectable Effect (MDE)"
    )
    subtitle = (
        f"Baseline CR = {power.baseline_cr * 100:.1f}% | alpha = {power.alpha:.2f} | "
        f"Required N/Variant = {power.required_sample_per_variant:,} ({power.estimated_duration_days} days)"
        if locale == "en"
        else f"CR Dasar = {power.baseline_cr * 100:.1f}% | alfa = {power.alpha:.2f} | "
        f"Sampel Dibutuhkan/Varian = {power.required_sample_per_variant:,} ({power.estimated_duration_days} hari)"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower right", fontsize=8.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_all_figures(
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    power: PowerAnalysisResult,
    output_dir: Path = Path("reports/figures"),
    locale: str = "en",
    seed: int = 42,
) -> dict[str, Path]:
    """Generate all four static PNG report figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "conversion_funnel": plot_conversion_funnel(
            frequentist, output_dir / "conversion_funnel.png", locale=locale
        ),
        "frequentist_ci_segments": plot_frequentist_segments(
            frequentist, output_dir / "frequentist_ci_segments.png", locale=locale
        ),
        "bayesian_posteriors": plot_bayesian_posteriors(
            bayesian, output_dir / "bayesian_posteriors.png", locale=locale, seed=seed
        ),
        "power_curve": plot_power_curve(
            power, output_dir / "power_curve.png", locale=locale
        ),
    }
