"""Internationalization and Babel locale formatting utilities."""

from __future__ import annotations

from typing import Literal, cast

from babel.numbers import format_currency, format_decimal, format_percent

LocaleCode = Literal["en", "id"]
SUPPORTED_LOCALES: tuple[LocaleCode, ...] = ("en", "id")

_TRANSLATIONS: dict[LocaleCode, dict[str, str]] = {
    "en": {
        "app_title": "Checkout Redesign A/B Experiment Analysis",
        "exec_summary_title": "Executive Summary",
        "experiment_overview": "Experiment Overview",
        "key_findings": "Key Findings",
        "methodology_notes": "Methodology & Statistical Rigor",
        "caveats_title": "Caveats & Operational Guardrails",
        "decision_label": "Recommendation",
        "decision_SHIP": "SHIP",
        "decision_HOLD": "HOLD",
        "decision_ITERATE": "ITERATE",
        "conversion_rate": "Conversion Rate",
        "relative_lift": "Relative Lift",
        "absolute_diff": "Absolute Difference",
        "p_value": "p-value",
        "ci_95": "95% Confidence Interval",
        "cohens_h": "Cohen's h",
        "cohens_d": "Cohen's d",
        "rpu": "Revenue per User (RPU)",
        "aov": "Average Order Value (AOV)",
        "prob_to_beat": "P(Treatment > Control)",
        "expected_loss": "Expected Loss",
        "hdi_95": "95% HDI",
        "rope": "ROPE",
        "srm_check": "Sample Ratio Mismatch (SRM) Check",
        "srm_passed": "PASSED",
        "srm_failed": "FAILED",
        "required_sample": "Required Sample / Variant",
        "achieved_power": "Achieved Power",
        "estimated_duration": "Estimated Duration (Days)",
        "data_retained_rate": "Retained Rate",
        "variant": "Variant",
        "control": "Control",
        "treatment": "Treatment",
        "users": "Users",
        "conversions": "Conversions",
        "rate": "Rate",
        "lift": "Lift",
        "stage": "Funnel Stage",
        "dimension": "Dimension",
        "segment": "Segment",
        "raw_p_value": "Raw p-value",
        "bh_p_value": "BH-Adjusted p-value",
        "significant": "Significant",
        "yes": "Yes",
        "no": "No",
        "prior_name": "Prior Specification",
        "prob_beats_control": "P(Treatment > Control)",
        "expected_loss_header": "Expected Loss (CR)",
        "rel_lift_mean": "Rel. Lift Mean",
        "metric": "Metric",
        "lift_evidence": "Lift / Evidence",
        "check": "Check",
        "value": "Value",
        "details": "Details",
        "event_cleaning": "Event Cleaning",
        "ctrl_cr": "Ctrl CR",
        "trt_cr": "Trt CR",
        "prior_sensitivity_title": "Bayesian Prior Sensitivity",
        "projected_30d_title": "30-Day Business Projection",
        "bayesian_posterior_label": "Bayesian Posterior",
        "bayesian_rope_row": "Bayesian P(T > C) / P(> ROPE)",
        "tab_overview": "Overview",
        "tab_data_quality": "Data Quality",
        "tab_power": "Power & Sample Size",
        "tab_frequentist": "Frequentist Inference",
        "tab_bayesian": "Bayesian Inference",
        "tab_segments": "Funnel & Segments",
        "tab_report": "Executive Report",
        "sidebar_config": "Experiment Parameters",
        "locale_selector": "Language / Locale",
        "run_analysis_btn": "Run Analysis",
        "download_json_btn": "Download JSON",
        "download_md_btn": "Download Executive Summary (MD)",
        "download_pdf_btn": "Download PDF Report",
        "exports_title": "Exports",
        "raw_clean_rows": "Clean / Raw Rows",
        "anomalies_removed": "Anomalies Removed",
        "total_sample_needed": "Total Sample Needed",
        "baseline_cr_label": "Baseline CR",
        "mde_rel_label": "Relative MDE",
        "alpha_label": "Significance Level (alpha)",
        "power_label": "Target Power",
        "prior_alpha_label": "Bayesian Prior Alpha",
        "prior_beta_label": "Bayesian Prior Beta",
        "daily_traffic_label": "Daily Traffic",
        "sig_abbr": "Sig.",
        "bh_sig_abbr": "BH Sig.",
        "ns_abbr": "NS",
        "mean_label": "Mean",
        "overall_label": "Overall",
        "target_power_label": "Target Power",
        "fig_funnel_title": "Conversion Funnel Analysis",
        "fig_segments_title": "Overall & Subgroup Lift (95% Confidence Intervals)",
        "fig_bayesian_title": "Bayesian Posterior Conversion & Relative Lift (95% HDI & ROPE)",
        "fig_power_title": "Power & Sample Size Analysis",
        "md_situation_label": "Situation",
        "md_complication_label": "Complication",
        "md_srm_label": "Validity Gate",
        "md_power_label": "Power Adequacy",
        "md_scqa_heading": "1. Situation & Data Quality (SCQA)",
        "md_impact_heading": "3. Quantified Business Impact",
        "md_rec_heading": "4. Recommendation & Operational Guardrails",
        "md_final_decision": "Final Decision",
        "md_segment_highlights": "Segment Highlights (BH-FDR)",
        "md_expected_loss_line": "Bayesian Expected Loss (choosing treatment)",
        "caveat_synthetic": "Dataset is deterministically simulated to mirror realistic checkout funnel and revenue behavior.",
        "caveat_srm": "SRM gate (alpha = 0.001) must pass before trusting downstream treatment effects.",
        "caveat_novelty": "Monitor post-launch metrics for novelty decay over at least two full weekly cycles.",
        "caveat_peeking": "Fixed-horizon frequentist p-values assume no unadjusted interim peeking prior to target sample size.",
        "caveat_fdr": "Segment p-values are adjusted using Benjamini-Hochberg FDR control across all evaluated subgroups.",
    },
    "id": {
        "app_title": "Analisis Eksperimen A/B Desain Ulang Checkout",
        "exec_summary_title": "Ringkasan Eksekutif",
        "experiment_overview": "Gambaran Umum Eksperimen",
        "key_findings": "Temuan Utama",
        "methodology_notes": "Metodologi & Ketelitian Statistik",
        "caveats_title": "Catatan Penting & Batasan Operasional",
        "decision_label": "Rekomendasi",
        "decision_SHIP": "RILIS",
        "decision_HOLD": "TAHAN",
        "decision_ITERATE": "ITERASI",
        "conversion_rate": "Tingkat Konversi",
        "relative_lift": "Peningkatan Relatif",
        "absolute_diff": "Selisih Absolut",
        "p_value": "Nilai-p",
        "ci_95": "Interval Kepercayaan 95%",
        "cohens_h": "Cohen's h",
        "cohens_d": "Cohen's d",
        "rpu": "Pendapatan per Pengguna (RPU)",
        "aov": "Nilai Pesanan Rata-rata (AOV)",
        "prob_to_beat": "P(Perlakuan > Kontrol)",
        "expected_loss": "Ekspektasi Kerugian",
        "hdi_95": "HDI 95%",
        "rope": "ROPE",
        "srm_check": "Pemeriksaan Sample Ratio Mismatch (SRM)",
        "srm_passed": "LOLOS",
        "srm_failed": "GAGAL",
        "required_sample": "Sampel Dibutuhkan / Varian",
        "achieved_power": "Daya Statistik Tercapai",
        "estimated_duration": "Estimasi Durasi (Hari)",
        "data_retained_rate": "Rasio Data Dipertahankan",
        "variant": "Varian",
        "control": "Kontrol",
        "treatment": "Perlakuan",
        "users": "Pengguna",
        "conversions": "Konversi",
        "rate": "Tingkat",
        "lift": "Peningkatan",
        "stage": "Tahap Funnel",
        "dimension": "Dimensi",
        "segment": "Segmen",
        "raw_p_value": "Nilai-p Mentah",
        "bh_p_value": "Nilai-p Koreksi BH",
        "significant": "Signifikan",
        "yes": "Ya",
        "no": "Tidak",
        "prior_name": "Spesifikasi Prior",
        "prob_beats_control": "P(Perlakuan > Kontrol)",
        "expected_loss_header": "Ekspektasi Kerugian (CR)",
        "rel_lift_mean": "Rata-rata Lift Rel.",
        "metric": "Metrik",
        "lift_evidence": "Selisih / Bukti Statistik",
        "check": "Pemeriksaan",
        "value": "Nilai",
        "details": "Detail",
        "event_cleaning": "Pembersihan Data",
        "ctrl_cr": "CR Kontrol",
        "trt_cr": "CR Perlakuan",
        "prior_sensitivity_title": "Analisis Sensitivitas Prior Bayesian",
        "projected_30d_title": "Proyeksi Bisnis 30 Hari",
        "bayesian_posterior_label": "Inferensi Bayesian",
        "bayesian_rope_row": "Bayesian P(T > C) / P(> ROPE)",
        "tab_overview": "Ringkasan",
        "tab_data_quality": "Kualitas Data",
        "tab_power": "Daya & Ukuran Sampel",
        "tab_frequentist": "Inferensi Frequentist",
        "tab_bayesian": "Inferensi Bayesian",
        "tab_segments": "Funnel & Segmen",
        "tab_report": "Laporan Eksekutif",
        "sidebar_config": "Parameter Eksperimen",
        "locale_selector": "Bahasa / Lokalisasi",
        "run_analysis_btn": "Jalankan Analisis",
        "download_json_btn": "Unduh JSON",
        "download_md_btn": "Unduh Ringkasan Eksekutif (MD)",
        "download_pdf_btn": "Unduh Laporan PDF",
        "exports_title": "Ekspor Laporan",
        "raw_clean_rows": "Baris Bersih / Mentah",
        "anomalies_removed": "Anomali Dihapus",
        "total_sample_needed": "Total Sampel Dibutuhkan",
        "baseline_cr_label": "CR Dasar",
        "mde_rel_label": "MDE Relatif",
        "alpha_label": "Tingkat Signifikansi (alfa)",
        "power_label": "Target Daya Statistik",
        "prior_alpha_label": "Prior Alpha Bayesian",
        "prior_beta_label": "Prior Beta Bayesian",
        "daily_traffic_label": "Trafik Harian",
        "sig_abbr": "Sig.",
        "bh_sig_abbr": "Sig. BH",
        "ns_abbr": "TS",
        "mean_label": "Rata-rata",
        "overall_label": "Keseluruhan",
        "target_power_label": "Target Daya",
        "fig_funnel_title": "Analisis Funnel Konversi",
        "fig_segments_title": "Peningkatan Populasi & Subkelompok (CI 95%)",
        "fig_bayesian_title": "Posterior Konversi & Lift Relatif Bayesian (HDI 95% & ROPE)",
        "fig_power_title": "Analisis Daya Statistik & Ukuran Sampel",
        "md_situation_label": "Situasi",
        "md_complication_label": "Komplikasi",
        "md_srm_label": "Gerbang Validitas",
        "md_power_label": "Kecukupan Daya",
        "md_scqa_heading": "1. Situasi & Kualitas Data (SCQA)",
        "md_impact_heading": "3. Proyeksi Dampak Bisnis",
        "md_rec_heading": "4. Rekomendasi & Batasan Operasional",
        "md_final_decision": "Keputusan Akhir",
        "md_segment_highlights": "Sorotan Segmen (BH-FDR)",
        "md_expected_loss_line": "Ekspektasi Kerugian Bayesian (memilih perlakuan)",
        "caveat_synthetic": "Dataset disimulasikan secara deterministik untuk mencerminkan perilaku funnel checkout dan pendapatan.",
        "caveat_srm": "Gerbang SRM (alfa = 0,001) harus lolos sebelum mempercayai estimasi efek perlakuan.",
        "caveat_novelty": "Pantau metrik pasca-peluncuran terhadap peluruhan efek kebaruan minimal dua siklus mingguan penuh.",
        "caveat_peeking": "Nilai-p frequentist horizon tetap mengasumsikan tidak ada pemeriksaan dini tanpa penyesuaian sebelum target sampel.",
        "caveat_fdr": "Nilai-p segmen disesuaikan menggunakan kontrol FDR Benjamini-Hochberg di seluruh subkelompok yang diuji.",
    },
}


def validate_locale(locale: str) -> LocaleCode:
    """Validate and return a supported locale code or raise ValueError."""
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale {locale!r}. Expected one of {SUPPORTED_LOCALES}."
        )
    return cast(LocaleCode, locale)


def _decimal_pattern(prefix: str, decimals: int, suffix: str = "") -> str:
    if decimals < 0:
        raise ValueError("decimals must be >= 0.")
    frac = f".{'0' * decimals}" if decimals > 0 else ""
    return f"{prefix}#,##0{frac}{suffix}"


def t(key: str, locale: str = "en", **kwargs: object) -> str:
    """Translate a key for the given locale and optionally format arguments."""
    loc = validate_locale(locale)
    template = _TRANSLATIONS[loc][key]
    return template.format(**kwargs) if kwargs else template


def fmt_num(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a decimal number using Babel locale rules."""
    loc = validate_locale(locale)
    pattern = _decimal_pattern("", decimals)
    return format_decimal(value, format=pattern, locale=loc)


def fmt_signed_num(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a decimal number with explicit '+' for positive values and '-' for negative."""
    formatted = fmt_num(value, locale=locale, decimals=decimals)
    return f"+{formatted}" if value > 0 else formatted


def fmt_pct(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a proportion (e.g. 0.1234 -> '12.34%') using Babel locale rules."""
    loc = validate_locale(locale)
    pattern = _decimal_pattern("", decimals, "%")
    return format_percent(value, format=pattern, locale=loc)


def fmt_signed_pct(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a proportion with explicit '+' for positive values."""
    formatted = fmt_pct(value, locale=locale, decimals=decimals)
    return f"+{formatted}" if value > 0 else formatted


def fmt_currency(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a USD currency amount using Babel locale rules."""
    loc = validate_locale(locale)
    pattern = _decimal_pattern("¤", decimals)
    return format_currency(
        value,
        "USD",
        format=pattern,
        locale=loc,
        currency_digits=(decimals > 0),
    )


def fmt_signed_currency(value: float, locale: str = "en", decimals: int = 2) -> str:
    """Format a USD currency amount with explicit '+' for positive values."""
    formatted = fmt_currency(value, locale=locale, decimals=decimals)
    return f"+{formatted}" if value > 0 else formatted


def fmt_pval(value: float, locale: str = "en", decimals: int = 4) -> str:
    """Format p-values, rendering non-zero sub-threshold values as '< 0.0001'."""
    loc = validate_locale(locale)
    threshold = 10.0 ** (-decimals)
    if 0.0 <= value < threshold:
        return f"< {fmt_num(threshold, locale=loc, decimals=decimals)}"
    return fmt_num(value, locale=loc, decimals=decimals)
