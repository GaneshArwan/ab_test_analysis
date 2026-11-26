"""Interactive Streamlit dashboard for Checkout Redesign A/B Experiment Analysis."""

from __future__ import annotations

import hashlib
import html
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.i18n import (
    SUPPORTED_LOCALES,
    fmt_currency,
    fmt_num,
    fmt_pct,
    fmt_pval,
    fmt_signed_currency,
    fmt_signed_num,
    fmt_signed_pct,
    t,
)
from src.power import run_power_analysis
from src.run_analysis import run_pipeline
from src.schemas import AnalysisReport, ExperimentConfig
from src.visualizations import plot_power_curve


@st.cache_data(show_spinner=False)
def _cached_run_pipeline(
    config_dict: dict[str, int | float],
    locale: str,
) -> tuple[str, dict[str, bytes], bytes, bytes]:
    cfg = ExperimentConfig(**config_dict)
    payload = json.dumps({"config": config_dict, "locale": locale}, sort_keys=True)
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    run_root = Path(tempfile.gettempdir()) / "ab_test_analysis_cache" / cache_key
    output_dir = run_root / "reports"
    data_path = run_root / "data" / "ab_test_data.csv"

    report, fig_paths = run_pipeline(
        config=cfg,
        locale=locale,
        output_dir=output_dir,
        data_path=data_path,
    )
    fig_bytes = {k: p.read_bytes() for k, p in fig_paths.items() if p.exists()}
    md_bytes = (output_dir / "executive_summary.md").read_bytes()
    pdf_bytes = (output_dir / "ab_test_report.pdf").read_bytes()
    return report.model_dump_json(), fig_bytes, md_bytes, pdf_bytes


def _decision_badge_html(report: AnalysisReport, locale: str) -> str:
    palette = {
        "SHIP": ("#DCFCE7", "#166534", "#15803D"),
        "HOLD": ("#FEE2E2", "#991B1B", "#B91C1C"),
        "ITERATE": ("#FEF3C7", "#92400E", "#B45309"),
    }
    bg, fg, border = palette[report.decision]
    label = html.escape(t(f"decision_{report.decision}", locale))
    code = html.escape(report.decision)
    title = html.escape(t("decision_label", locale))
    reason = html.escape(report.recommendation_reason)
    return (
        f'<div role="status" style="background:{bg};border:1px solid {border};'
        f'border-left:6px solid {border};border-radius:8px;padding:14px 18px;margin-bottom:16px;">'
        f'<div style="font-size:1.1rem;font-weight:700;color:{fg};margin-bottom:6px;">'
        f'{title}: {label} ({code})</div>'
        f'<div style="font-size:0.96rem;color:#0F172A;line-height:1.45;">{reason}</div>'
        f"</div>"
    )


def _render_sidebar(locale_default: str) -> tuple[str, ExperimentConfig | None]:
    locale = st.sidebar.selectbox(
        t("locale_selector", locale_default),
        options=list(SUPPORTED_LOCALES),
        format_func=lambda loc: "English (en)" if loc == "en" else "Bahasa Indonesia (id)",
    )
    st.sidebar.header(t("sidebar_config", locale))

    seed = int(
        st.sidebar.number_input(
            "Seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            step=1,
        )
    )
    n_users = int(
        st.sidebar.slider(
            "Simulated Users" if locale == "en" else "Jumlah Pengguna Simulasi",
            min_value=1000,
            max_value=200000,
            value=40000,
            step=1000,
        )
    )
    baseline_cr = float(
        st.sidebar.slider(
            t("baseline_cr_label", locale),
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            format="%.2f",
        )
    )
    mde_relative = float(
        st.sidebar.slider(
            t("mde_rel_label", locale),
            min_value=0.02,
            max_value=1.00,
            value=0.12,
            step=0.01,
            format="%.2f",
        )
    )
    alpha = float(
        st.sidebar.slider(
            t("alpha_label", locale),
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
        )
    )
    power = float(
        st.sidebar.slider(
            t("power_label", locale),
            min_value=0.60,
            max_value=0.99,
            value=0.80,
            step=0.01,
            format="%.2f",
        )
    )
    prior_alpha = float(
        st.sidebar.number_input(
            t("prior_alpha_label", locale),
            min_value=0.5,
            max_value=500.0,
            value=10.0,
            step=1.0,
        )
    )
    prior_beta = float(
        st.sidebar.number_input(
            t("prior_beta_label", locale),
            min_value=0.5,
            max_value=500.0,
            value=90.0,
            step=1.0,
        )
    )
    rope_margin = float(
        st.sidebar.slider(
            "ROPE Margin (Abs. CR)" if locale == "en" else "Margin ROPE (CR Absolut)",
            min_value=0.000,
            max_value=0.050,
            value=0.005,
            step=0.001,
            format="%.3f",
        )
    )
    daily_traffic = int(
        st.sidebar.number_input(
            t("daily_traffic_label", locale),
            min_value=100,
            max_value=1000000,
            value=3000,
            step=100,
        )
    )

    try:
        cfg = ExperimentConfig(
            seed=seed,
            n_users=n_users,
            alpha=alpha,
            power=power,
            baseline_cr=baseline_cr,
            mde_relative=mde_relative,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            rope_margin=rope_margin,
            daily_traffic=daily_traffic,
        )
    except ValidationError:
        msg = (
            "Invalid experiment parameters. Ensure baseline CR * (1 + MDE) < 100% and numeric ranges are valid."
            if locale == "en"
            else "Parameter eksperimen tidak valid. Pastikan CR dasar * (1 + MDE) < 100% dan rentang angka sesuai."
        )
        st.sidebar.error(msg)
        return locale, None

    return locale, cfg


def _render_executive_tab(
    report: AnalysisReport,
    report_json: str,
    md_bytes: bytes,
    pdf_bytes: bytes,
    locale: str,
) -> None:
    st.markdown(_decision_badge_html(report, locale), unsafe_allow_html=True)

    freq = report.frequentist
    bayes = report.bayesian
    dq = report.data_quality
    srm = report.srm

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        t("ctrl_cr", locale),
        fmt_pct(freq.control_cr, locale, 2),
        f"n = {fmt_num(freq.control_n, locale, 0)}",
    )
    col2.metric(
        t("trt_cr", locale),
        fmt_pct(freq.treatment_cr, locale, 2),
        f"{fmt_signed_pct(freq.rel_lift, locale, 2)} rel.",
    )
    col3.metric(
        t("prob_to_beat", locale),
        fmt_pct(bayes.prob_treatment_beats_control, locale, 2),
        f"P(> ROPE): {fmt_pct(bayes.prob_above_rope, locale, 1)}",
    )
    col4.metric(
        t("rpu", locale),
        fmt_currency(freq.treatment_rpu, locale, 2),
        fmt_signed_currency(freq.rpu_diff, locale, 2),
    )

    st.subheader(f"{t('tab_data_quality', locale)} & {t('srm_check', locale)}")
    dq_cols = st.columns(4)
    dq_cols[0].metric(
        t("raw_clean_rows", locale),
        f"{fmt_num(dq.clean_rows, locale, 0)} / {fmt_num(dq.raw_rows, locale, 0)}",
    )
    dq_cols[1].metric(t("data_retained_rate", locale), fmt_pct(dq.retained_rate, locale, 2))
    dq_cols[2].metric(
        t("anomalies_removed", locale),
        fmt_num(
            dq.duplicates_removed
            + dq.mixed_group_users_removed
            + dq.invalid_revenue_removed
            + dq.invalid_funnel_removed,
            locale,
            0,
        ),
    )
    dq_cols[3].metric(
        t("srm_check", locale),
        t("srm_passed", locale) if srm.passed else t("srm_failed", locale),
        f"p = {fmt_pval(srm.p_value, locale, 4)}",
    )

    st.subheader(t("exports_title", locale))
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button(
        t("download_json_btn", locale),
        data=report_json.encode("utf-8"),
        file_name="analysis_results.json",
        mime="application/json",
    )
    dl2.download_button(
        t("download_md_btn", locale),
        data=md_bytes,
        file_name="executive_summary.md",
        mime="text/markdown",
    )
    dl3.download_button(
        t("download_pdf_btn", locale),
        data=pdf_bytes,
        file_name="ab_test_report.pdf",
        mime="application/pdf",
    )

    st.subheader(t("caveats_title", locale))
    for key in ("caveat_srm", "caveat_fdr", "caveat_novelty", "caveat_peeking", "caveat_synthetic"):
        st.markdown(f"- {t(key, locale)}")


def _render_inference_tab(
    report: AnalysisReport,
    fig_bytes: dict[str, bytes],
    locale: str,
) -> None:
    freq = report.frequentist
    bayes = report.bayesian

    left, right = st.columns(2)
    with left:
        st.subheader(t("tab_frequentist", locale))
        st.metric(t("absolute_diff", locale), f"{fmt_signed_num(freq.abs_diff * 100.0, locale, 2)} pp")
        st.metric(
            t("ci_95", locale),
            f"[{fmt_signed_num(freq.diff_ci_low * 100.0, locale, 2)}, {fmt_signed_num(freq.diff_ci_high * 100.0, locale, 2)}] pp",
        )
        st.metric(t("p_value", locale), fmt_pval(freq.z_p_value, locale, 4))
        st.metric(t("cohens_h", locale), fmt_num(freq.cohens_h, locale, 4))
        st.metric(
            t("aov", locale),
            f"{fmt_currency(freq.control_aov, locale, 2)} -> {fmt_currency(freq.treatment_aov, locale, 2)}",
        )

    with right:
        st.subheader(t("tab_bayesian", locale))
        st.metric(t("prob_to_beat", locale), fmt_pct(bayes.prob_treatment_beats_control, locale, 2))
        st.metric(
            f"{t('hdi_95', locale)} ({t('relative_lift', locale)})",
            f"[{fmt_signed_pct(bayes.rel_lift_hdi_low, locale, 2)}, {fmt_signed_pct(bayes.rel_lift_hdi_high, locale, 2)}]",
        )
        st.metric(t("expected_loss", locale), fmt_pct(bayes.expected_loss_treatment, locale, 4))
        st.metric("P(> ROPE)", fmt_pct(bayes.prob_above_rope, locale, 2))
        st.metric("P(in ROPE)", fmt_pct(bayes.prob_in_rope, locale, 2))

    if "bayesian_posteriors" in fig_bytes:
        st.image(fig_bytes["bayesian_posteriors"], use_container_width=True)

    st.subheader(t("prior_sensitivity_title", locale))
    sens_df = pd.DataFrame(
        [
            {
                t("prior_name", locale): s.prior_name,
                t("prob_beats_control", locale): fmt_pct(s.prob_treatment_beats_control, locale, 2),
                t("expected_loss_header", locale): fmt_pct(s.expected_loss_treatment, locale, 4),
                t("relative_lift", locale): fmt_signed_pct(s.rel_lift_mean, locale, 2),
                t("hdi_95", locale): f"[{fmt_signed_pct(s.hdi_low, locale, 2)}, {fmt_signed_pct(s.hdi_high, locale, 2)}]",
            }
            for s in bayes.sensitivity
        ]
    )
    st.dataframe(sens_df, use_container_width=True, hide_index=True)


def _render_funnel_segments_tab(
    report: AnalysisReport,
    fig_bytes: dict[str, bytes],
    locale: str,
) -> None:
    freq = report.frequentist

    if "conversion_funnel" in fig_bytes:
        st.image(fig_bytes["conversion_funnel"], use_container_width=True)

    funnel_df = pd.DataFrame(
        [
            {
                t("stage", locale): stage.stage,
                f"{t('control', locale)} {t('users', locale)}": stage.control_n,
                f"{t('control', locale)} {t('rate', locale)}": fmt_pct(stage.control_rate, locale, 2),
                f"{t('treatment', locale)} {t('users', locale)}": stage.treatment_n,
                f"{t('treatment', locale)} {t('rate', locale)}": fmt_pct(stage.treatment_rate, locale, 2),
                t("relative_lift", locale): fmt_signed_pct(stage.relative_lift, locale, 2),
                t("p_value", locale): fmt_pval(stage.p_value, locale, 4),
            }
            for stage in freq.funnel_stages
        ]
    )
    st.dataframe(funnel_df, use_container_width=True, hide_index=True)

    if "frequentist_ci_segments" in fig_bytes:
        st.image(fig_bytes["frequentist_ci_segments"], use_container_width=True)

    seg_df = pd.DataFrame(
        [
            {
                t("dimension", locale): s.dimension.title(),
                t("segment", locale): s.segment.replace("_", " ").title(),
                t("ctrl_cr", locale): fmt_pct(s.control_cr, locale, 2),
                t("trt_cr", locale): fmt_pct(s.treatment_cr, locale, 2),
                t("relative_lift", locale): fmt_signed_pct(s.rel_lift, locale, 2),
                t("raw_p_value", locale): fmt_pval(s.p_value, locale, 4),
                t("bh_p_value", locale): fmt_pval(s.p_value_bh, locale, 4),
                t("significant", locale): t("yes", locale) if s.significant_bh else t("no", locale),
            }
            for s in freq.segments
        ]
    )
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


def _render_power_calculator_tab(report: AnalysisReport, locale: str) -> None:
    st.subheader(t("tab_power", locale))
    c1, c2, c3, c4, c5 = st.columns(5)
    calc_baseline = float(
        c1.number_input(
            t("baseline_cr_label", locale),
            min_value=0.01,
            max_value=0.90,
            value=float(report.config.baseline_cr),
            step=0.01,
            key="calc_baseline",
        )
    )
    calc_mde = float(
        c2.number_input(
            t("mde_rel_label", locale),
            min_value=0.01,
            max_value=2.00,
            value=float(report.config.mde_relative),
            step=0.01,
            key="calc_mde",
        )
    )
    calc_alpha = float(
        c3.number_input(
            t("alpha_label", locale),
            min_value=0.005,
            max_value=0.25,
            value=float(report.config.alpha),
            step=0.005,
            key="calc_alpha",
        )
    )
    calc_power = float(
        c4.number_input(
            t("power_label", locale),
            min_value=0.55,
            max_value=0.99,
            value=float(report.config.power),
            step=0.01,
            key="calc_power",
        )
    )
    calc_traffic = int(
        c5.number_input(
            t("daily_traffic_label", locale),
            min_value=10,
            max_value=1000000,
            value=int(report.config.daily_traffic),
            step=100,
            key="calc_traffic",
        )
    )

    try:
        calc_cfg = ExperimentConfig(
            baseline_cr=calc_baseline,
            mde_relative=calc_mde,
            alpha=calc_alpha,
            power=calc_power,
            daily_traffic=calc_traffic,
        )
        calc_res = run_power_analysis(
            calc_cfg,
            actual_control_n=report.frequentist.control_n,
            actual_treatment_n=report.frequentist.treatment_n,
        )
    except (ValidationError, ValueError):
        st.error(
            "Invalid power calculator inputs. Ensure target CR remains below 100%."
            if locale == "en"
            else "Input kalkulator daya tidak valid. Pastikan target CR di bawah 100%."
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("required_sample", locale), fmt_num(calc_res.required_sample_per_variant, locale, 0))
    m2.metric(t("total_sample_needed", locale), fmt_num(calc_res.total_required_sample, locale, 0))
    m3.metric(t("estimated_duration", locale), str(calc_res.estimated_duration_days))
    m4.metric(t("achieved_power", locale), fmt_pct(calc_res.achieved_power, locale, 2))

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = Path(tmp_dir) / "power_calculator.png"
        plot_power_curve(calc_res, fig_path, locale=locale)
        if fig_path.exists():
            st.image(fig_path.read_bytes(), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Checkout Redesign A/B Experiment Analysis",
        layout="wide",
    )

    locale, cfg = _render_sidebar("en")
    st.title(t("app_title", locale))

    if cfg is None:
        return

    try:
        report_json, fig_bytes, md_bytes, pdf_bytes = _cached_run_pipeline(cfg.model_dump(), locale)
        report = AnalysisReport.model_validate_json(report_json)
    except (ValidationError, ValueError, OSError, RuntimeError):
        st.error(
            "Unable to complete analysis with the current configuration. Please adjust inputs and retry."
            if locale == "en"
            else "Analisis tidak dapat diselesaikan dengan konfigurasi saat ini. Sesuaikan parameter dan coba lagi."
        )
        return

    tab_exec, tab_inf, tab_seg, tab_pwr = st.tabs(
        [
            t("exec_summary_title", locale),
            f"{t('tab_frequentist', locale)} vs {t('tab_bayesian', locale)}",
            t("tab_segments", locale),
            t("tab_power", locale),
        ]
    )

    with tab_exec:
        _render_executive_tab(report, report_json, md_bytes, pdf_bytes, locale)
    with tab_inf:
        _render_inference_tab(report, fig_bytes, locale)
    with tab_seg:
        _render_funnel_segments_tab(report, fig_bytes, locale)
    with tab_pwr:
        _render_power_calculator_tab(report, locale)


if __name__ == "__main__":
    main()
