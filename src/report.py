"""Decision engine, SCQA executive summary generator, and ReportLab PDF exporter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.i18n import (
    fmt_currency,
    fmt_num,
    fmt_pct,
    fmt_pval,
    fmt_signed_currency,
    fmt_signed_num,
    fmt_signed_pct,
    t,
    validate_locale,
)
from src.schemas import (
    AnalysisReport,
    BayesianResult,
    DataQualitySummary,
    ExperimentConfig,
    FrequentistResult,
    PowerAnalysisResult,
    SRMResult,
)

_NAVY = colors.HexColor("#0F172A")
_BLUE = colors.HexColor("#1E3A8A")
_SLATE = colors.HexColor("#475569")
_LIGHT_BG = colors.HexColor("#F8FAFC")
_BORDER = colors.HexColor("#CBD5E1")
_SHIP_BG = colors.HexColor("#DCFCE7")
_SHIP_FG = colors.HexColor("#166534")
_HOLD_BG = colors.HexColor("#FEE2E2")
_HOLD_FG = colors.HexColor("#991B1B")
_ITERATE_BG = colors.HexColor("#FEF3C7")
_ITERATE_FG = colors.HexColor("#92400E")


def _determine_decision(
    srm: SRMResult,
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
) -> Literal["SHIP", "HOLD", "ITERATE"]:
    if not srm.passed:
        return "HOLD"
    if (
        frequentist.significant
        and bayesian.prob_treatment_beats_control >= 0.95
        and bayesian.prob_above_rope >= 0.90
    ):
        return "SHIP"
    return "ITERATE"


def _build_recommendation_reason(
    decision: Literal["SHIP", "HOLD", "ITERATE"],
    srm: SRMResult,
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    locale: str,
) -> str:
    ctrl_cr = fmt_pct(frequentist.control_cr, locale, 2)
    trt_cr = fmt_pct(frequentist.treatment_cr, locale, 2)
    rel_lift = fmt_signed_pct(frequentist.rel_lift, locale, 2)
    p_val = fmt_pval(frequentist.z_p_value, locale, 4)
    prob_beat = fmt_pct(bayesian.prob_treatment_beats_control, locale, 2)
    prob_rope = fmt_pct(bayesian.prob_above_rope, locale, 2)
    rpu_diff = fmt_signed_currency(frequentist.rpu_diff, locale, 2)

    if decision == "HOLD":
        srm_p = fmt_pval(srm.p_value, locale, 4)
        if locale == "id":
            return (
                f"Tahan peluncuran karena pemeriksaan Sample Ratio Mismatch (SRM) gagal "
                f"(p = {srm_p}; kontrol {srm.control_n:,} vs perlakuan {srm.treatment_n:,}). "
                f"Alokasi trafik harus diaudit sebelum mempercayai estimasi efek."
            )
        return (
            f"Hold rollout because the Sample Ratio Mismatch (SRM) gate failed "
            f"(p = {srm_p}; control {srm.control_n:,} vs. treatment {srm.treatment_n:,}). "
            f"Traffic assignment must be audited before trusting treatment estimates."
        )

    if decision == "SHIP":
        if locale == "id":
            return (
                f"Rilis desain ulang checkout: tingkat konversi naik dari {ctrl_cr} menjadi {trt_cr} "
                f"({rel_lift} relatif, p = {p_val}), dengan probabilitas Bayesian mengungguli kontrol "
                f"{prob_beat}, probabilitas di atas ROPE {prob_rope}, dan dampak RPU {rpu_diff} per pengguna."
            )
        return (
            f"Ship the checkout redesign: conversion rate rose from {ctrl_cr} to {trt_cr} "
            f"({rel_lift} relative lift, p = {p_val}), backed by {prob_beat} Bayesian probability "
            f"of beating control, {prob_rope} probability above the practical significance threshold (ROPE), "
            f"and {rpu_diff} incremental revenue per user."
        )

    if locale == "id":
        return (
            f"Lakukan iterasi sebelum rilis penuh: tingkat konversi berubah dari {ctrl_cr} menjadi {trt_cr} "
            f"({rel_lift} relatif, p = {p_val}; P(Perlakuan > Kontrol) = {prob_beat}; "
            f"P(> ROPE) = {prob_rope}), belum memenuhi seluruh ambang kelayakan rilis."
        )
    return (
        f"Iterate before full rollout: conversion moved from {ctrl_cr} to {trt_cr} "
        f"({rel_lift} relative lift, p = {p_val}; P(Treatment > Control) = {prob_beat}; "
        f"P(> ROPE) = {prob_rope}), falling short of all ship thresholds simultaneously."
    )


def build_analysis_report(
    config: ExperimentConfig,
    data_quality: DataQualitySummary,
    power_analysis: PowerAnalysisResult,
    srm: SRMResult,
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    locale: str = "en",
    generated_at: datetime | None = None,
) -> AnalysisReport:
    """Assemble validated AnalysisReport with deterministic Ship/Hold/Iterate decision."""
    loc = validate_locale(locale)
    decision = _determine_decision(srm=srm, frequentist=frequentist, bayesian=bayesian)
    reason = _build_recommendation_reason(
        decision=decision,
        srm=srm,
        frequentist=frequentist,
        bayesian=bayesian,
        locale=loc,
    )
    return AnalysisReport(
        generated_at=generated_at or datetime.now(UTC),
        locale=loc,
        config=config,
        data_quality=data_quality,
        power_analysis=power_analysis,
        srm=srm,
        frequentist=frequentist,
        bayesian=bayesian,
        decision=decision,
        recommendation_reason=reason,
    )


def _projected_monthly_impact(report: AnalysisReport) -> tuple[float, float]:
    monthly_users = report.config.daily_traffic * 30.0
    inc_conversions = monthly_users * report.frequentist.abs_diff
    inc_revenue = monthly_users * report.frequentist.rpu_diff
    return inc_conversions, inc_revenue


def _highlight_segments(report: AnalysisReport) -> tuple[str, str]:
    sig_segs = [s for s in report.frequentist.segments if s.significant_bh]
    if not sig_segs:
        if report.locale == "id":
            return (
                "Tidak ada segmen yang signifikan secara statistik setelah koreksi BH-FDR.",
                "Periksa segmen dengan ukuran sampel lebih kecil pada iterasi berikutnya.",
            )
        return (
            "No individual subgroup reached statistical significance after BH-FDR correction.",
            "Collect additional volume before making subgroup-specific rollouts.",
        )

    best = max(sig_segs, key=lambda s: s.rel_lift)
    weakest = min(report.frequentist.segments, key=lambda s: s.rel_lift)
    best_str = (
        f"{best.dimension.title()}: {best.segment.replace('_', ' ').title()} "
        f"({fmt_signed_pct(best.rel_lift, report.locale, 1)} rel., "
        f"BH p = {fmt_pval(best.p_value_bh, report.locale, 4)})"
    )
    weak_str = (
        f"{weakest.dimension.title()}: {weakest.segment.replace('_', ' ').title()} "
        f"({fmt_signed_pct(weakest.rel_lift, report.locale, 1)} rel., "
        f"BH p = {fmt_pval(weakest.p_value_bh, report.locale, 4)})"
    )
    return best_str, weak_str


def write_executive_summary_md(report: AnalysisReport, output_path: Path) -> Path:
    """Write a SCQA / Pyramid Principle executive summary Markdown report."""
    loc = validate_locale(report.locale)
    freq = report.frequentist
    bayes = report.bayesian
    dq = report.data_quality
    pwr = report.power_analysis
    srm = report.srm
    dec_label = t(f"decision_{report.decision}", loc)
    inc_conv, inc_rev = _projected_monthly_impact(report)
    best_seg, weak_seg = _highlight_segments(report)

    srm_status = t("srm_passed", loc) if srm.passed else t("srm_failed", loc)
    ctrl_ci = f"[{fmt_pct(freq.control_ci_low, loc, 2)}, {fmt_pct(freq.control_ci_high, loc, 2)}]"
    trt_ci = f"[{fmt_pct(freq.treatment_ci_low, loc, 2)}, {fmt_pct(freq.treatment_ci_high, loc, 2)}]"
    diff_pp = fmt_signed_num(freq.abs_diff * 100.0, loc, 2)
    diff_ci_pp = (
        f"[{fmt_signed_num(freq.diff_ci_low * 100.0, loc, 2)}, "
        f"{fmt_signed_num(freq.diff_ci_high * 100.0, loc, 2)}] pp"
    )
    rel_hdi = (
        f"[{fmt_signed_pct(bayes.rel_lift_hdi_low, loc, 2)}, "
        f"{fmt_signed_pct(bayes.rel_lift_hdi_high, loc, 2)}]"
    )

    if loc == "id":
        meta_line = f"Dibuat pada: `{report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}` | Lokalisasi: `{loc}`"
        situation_body = (
            f"Pengalaman checkout yang dirancang ulang diuji terhadap aliran kontrol selama 14 hari pada "
            f"{fmt_num(dq.clean_rows, loc, 0)} pengguna bersih ({fmt_num(freq.control_n, loc, 0)} "
            f"{t('control', loc)}, {fmt_num(freq.treatment_n, loc, 0)} {t('treatment', loc)})."
        )
        complication_body = (
            f"Log mentah berisi {fmt_num(dq.raw_rows, loc, 0)} baris dengan anomali "
            f"(duplikat: {dq.duplicates_removed}, pengguna grup campuran: {dq.mixed_group_users_removed}, "
            f"pendapatan tidak valid: {dq.invalid_revenue_removed}, funnel rusak: {dq.invalid_funnel_removed}). "
            f"Setelah pembersihan ketat berbasis skema, {fmt_pct(dq.retained_rate, loc, 2)} data dipertahankan."
        )
        power_body = (
            f"Target sampel = {fmt_num(pwr.required_sample_per_variant, loc, 0)}/varian "
            f"(MDE relatif {fmt_pct(pwr.mde_relative, loc, 1)}, estimasi {pwr.estimated_duration_days} hari); "
            f"daya tercapai = **{fmt_pct(pwr.achieved_power, loc, 2)}**."
        )
        seg_line = f"Terkuat: {best_seg}. Terlemah: {weak_seg}."
        loss_unit = "poin konversi."
        impact_intro = (
            f"Berdasarkan trafik harian {fmt_num(report.config.daily_traffic, loc, 0)} pengguna "
            f"(~{fmt_num(report.config.daily_traffic * 30, loc, 0)} pengguna per 30 hari):"
        )
        conv_proj_line = f"**Proyeksi Konversi Tambahan (30 Hari):** {fmt_signed_num(inc_conv, loc, 0)} konversi/bulan."
        rev_proj_line = f"**Proyeksi Pendapatan Tambahan (30 Hari):** {fmt_signed_currency(inc_rev, loc, 0)}/bulan."
        diff_word = "Selisih"
    else:
        meta_line = f"Generated: `{report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}` | Locale: `{loc}`"
        situation_body = (
            f"A redesigned checkout flow was evaluated against the existing control experience "
            f"over a 14-day window across {fmt_num(dq.clean_rows, loc, 0)} validated users "
            f"({fmt_num(freq.control_n, loc, 0)} {t('control', loc)}, "
            f"{fmt_num(freq.treatment_n, loc, 0)} {t('treatment', loc)})."
        )
        complication_body = (
            f"Raw event logs contained {fmt_num(dq.raw_rows, loc, 0)} rows with tracking anomalies "
            f"(duplicates: {dq.duplicates_removed}, mixed-group users: {dq.mixed_group_users_removed}, "
            f"invalid revenue: {dq.invalid_revenue_removed}, broken funnel state: {dq.invalid_funnel_removed}). "
            f"Strict schema enforcement retained {fmt_pct(dq.retained_rate, loc, 2)} of raw events."
        )
        power_body = (
            f"Required sample = {fmt_num(pwr.required_sample_per_variant, loc, 0)}/variant "
            f"(target relative MDE {fmt_pct(pwr.mde_relative, loc, 1)}, estimated {pwr.estimated_duration_days} days); "
            f"achieved power = **{fmt_pct(pwr.achieved_power, loc, 2)}**."
        )
        seg_line = f"Strongest: {best_seg}. Weakest: {weak_seg}."
        loss_unit = "conversion rate risk."
        impact_intro = (
            f"Scaled to baseline site traffic of {fmt_num(report.config.daily_traffic, loc, 0)} daily users "
            f"(~{fmt_num(report.config.daily_traffic * 30, loc, 0)} users per 30-day month):"
        )
        conv_proj_line = f"**Projected Incremental Conversions (30 Days):** {fmt_signed_num(inc_conv, loc, 0)} conversions/month."
        rev_proj_line = f"**Projected Incremental Revenue (30 Days):** {fmt_signed_currency(inc_rev, loc, 0)}/month."
        diff_word = "Diff"

    lines = [
        f"# {t('app_title', loc)} — {t('exec_summary_title', loc)}",
        "",
        f"**{t('decision_label', loc)}: {dec_label} (`{report.decision}`)**  ",
        meta_line,
        "",
        f"> {report.recommendation_reason}",
        "",
        f"## {t('md_scqa_heading', loc)}",
        "",
        f"- **{t('md_situation_label', loc)}:** {situation_body}",
        f"- **{t('md_complication_label', loc)}:** {complication_body}",
        (
            f"- **{t('md_srm_label', loc)} ({t('srm_check', loc)}):** **{srm_status}** "
            f"(chi2 = {fmt_num(srm.chi2_stat, loc, 4)}, p = {fmt_pval(srm.p_value, loc, 4)}, "
            f"alpha = {fmt_num(srm.alpha, loc, 3)})."
        ),
        f"- **{t('md_power_label', loc)}:** {power_body}",
        "",
        f"## 2. {t('key_findings', loc)}",
        "",
        f"| {t('metric', loc)} | {t('control', loc)} | {t('treatment', loc)} | {t('lift_evidence', loc)} |",
        "|---|---|---|---|",
        (
            f"| {t('conversion_rate', loc)} | {fmt_pct(freq.control_cr, loc, 2)} {ctrl_ci} | "
            f"{fmt_pct(freq.treatment_cr, loc, 2)} {trt_ci} | "
            f"{diff_pp} pp ({fmt_signed_pct(freq.rel_lift, loc, 2)} rel.), CI {diff_ci_pp}, p = {fmt_pval(freq.z_p_value, loc, 4)} |"
        ),
        (
            f"| {t('bayesian_posterior_label', loc)} | {t('mean_label', loc)} {fmt_pct(bayes.control_post_mean, loc, 2)} | "
            f"{t('mean_label', loc)} {fmt_pct(bayes.treatment_post_mean, loc, 2)} | "
            f"P(T > C) = {fmt_pct(bayes.prob_treatment_beats_control, loc, 2)}, "
            f"{t('hdi_95', loc)} {rel_hdi}, P(> ROPE) = {fmt_pct(bayes.prob_above_rope, loc, 2)} |"
        ),
        (
            f"| {t('rpu', loc)} | {fmt_currency(freq.control_rpu, loc, 2)} | "
            f"{fmt_currency(freq.treatment_rpu, loc, 2)} | "
            f"{diff_word} {fmt_signed_currency(freq.rpu_diff, loc, 2)} "
            f"(Welch p = {fmt_pval(freq.rpu_welch_p, loc, 4)}, Mann-Whitney p = {fmt_pval(freq.rpu_mannwhitney_p, loc, 4)}) |"
        ),
        (
            f"| {t('aov', loc)} | {fmt_currency(freq.control_aov, loc, 2)} | "
            f"{fmt_currency(freq.treatment_aov, loc, 2)} | "
            f"{diff_word} {fmt_signed_currency(freq.treatment_aov - freq.control_aov, loc, 2)} |"
        ),
        "",
        f"- **{t('md_segment_highlights', loc)}:** {seg_line}",
        f"- **{t('md_expected_loss_line', loc)}:** {fmt_pct(bayes.expected_loss_treatment, loc, 4)} {loss_unit}",
        "",
        f"## {t('md_impact_heading', loc)}",
        "",
        impact_intro,
        f"- {conv_proj_line}",
        f"- {rev_proj_line}",
        "",
        f"## {t('md_rec_heading', loc)}",
        "",
        f"**{t('md_final_decision', loc)}:** **{dec_label} (`{report.decision}`)** — {report.recommendation_reason}",
        "",
        f"- {t('caveat_srm', loc)}",
        f"- {t('caveat_fdr', loc)}",
        f"- {t('caveat_novelty', loc)}",
        f"- {t('caveat_peeking', loc)}",
        f"- {t('caveat_synthetic', loc)}",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _styled_table(
    data: list[list[str]],
    col_widths: list[float],
    align_right_cols: tuple[int, ...] = (),
) -> Table:
    """Build a ReportLab table with wrapped Paragraph cells so text never overflows."""
    header_left = ParagraphStyle(
        "TblHeaderLeft",
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.white,
        alignment=TA_LEFT,
    )
    header_right = ParagraphStyle(
        "TblHeaderRight",
        parent=header_left,
        alignment=TA_RIGHT,
    )
    body_left = ParagraphStyle(
        "TblBodyLeft",
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.0,
        textColor=_NAVY,
        alignment=TA_LEFT,
    )
    body_right = ParagraphStyle(
        "TblBodyRight",
        parent=body_left,
        alignment=TA_RIGHT,
    )

    wrapped_rows: list[list[Paragraph]] = []
    for row_idx, row in enumerate(data):
        wrapped_row: list[Paragraph] = []
        for col_idx, cell in enumerate(row):
            is_right = col_idx in align_right_cols
            if row_idx == 0:
                style = header_right if is_right else header_left
            else:
                style = body_right if is_right else body_left
            wrapped_row.append(Paragraph(xml_escape(str(cell)), style))
        wrapped_rows.append(wrapped_row)

    table = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_BG]),
                ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _scaled_image(img_path: Path, max_width: float, max_height: float) -> Image:
    iw, ih = ImageReader(str(img_path)).getSize()
    scale = min(max_width / iw, max_height / ih)
    return Image(str(img_path), width=iw * scale, height=ih * scale)


def write_pdf_report(
    report: AnalysisReport,
    figure_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    """Export a formatted multi-page PDF executive & statistical report via ReportLab."""
    loc = validate_locale(report.locale)
    freq = report.frequentist
    bayes = report.bayesian
    dq = report.data_quality
    pwr = report.power_analysis
    srm = report.srm
    inc_conv, inc_rev = _projected_monthly_impact(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=t("app_title", loc),
    )

    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=_NAVY,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=_SLATE,
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=_BLUE,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextClean",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=_NAVY,
        spaceAfter=6,
    )

    story: list[object] = []
    story.append(Paragraph(xml_escape(t("app_title", loc)), title_style))
    meta_line = (
        f"{t('exec_summary_title', loc)} | "
        f"{report.generated_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Seed: {report.config.seed} | Locale: {loc}"
    )
    story.append(Paragraph(xml_escape(meta_line), subtitle_style))

    banner_colors = {
        "SHIP": (_SHIP_BG, _SHIP_FG),
        "HOLD": (_HOLD_BG, _HOLD_FG),
        "ITERATE": (_ITERATE_BG, _ITERATE_FG),
    }
    bg_col, fg_col = banner_colors[report.decision]
    dec_label = t(f"decision_{report.decision}", loc)
    banner_data = [
        [
            Paragraph(
                f"<b>{xml_escape(t('decision_label', loc))}: {xml_escape(dec_label)} ({report.decision})</b>",
                ParagraphStyle("BannerHead", parent=body_style, textColor=fg_col, fontSize=11, spaceAfter=2),
            )
        ],
        [
            Paragraph(
                xml_escape(report.recommendation_reason),
                ParagraphStyle("BannerBody", parent=body_style, textColor=_NAVY, fontSize=9.2, spaceAfter=0),
            )
        ],
    ]
    banner_table = Table(banner_data, colWidths=[7.2 * inch])
    banner_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_col),
                ("BOX", (0, 0), (-1, -1), 1.0, fg_col),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(banner_table)
    story.append(Spacer(1, 10))

    # Core KPI Summary Table (total width = 7.2 in).
    story.append(Paragraph(xml_escape(t("key_findings", loc)), h2_style))
    conv_unit = "konv." if loc == "id" else "conv."
    mo_unit = "/bln" if loc == "id" else "/mo"
    users_unit = t("users", loc).lower()
    kpi_rows = [
        [t("metric", loc), t("control", loc), t("treatment", loc), t("lift_evidence", loc)],
        [
            t("conversion_rate", loc),
            fmt_pct(freq.control_cr, loc, 2),
            fmt_pct(freq.treatment_cr, loc, 2),
            f"{fmt_signed_pct(freq.rel_lift, loc, 2)} rel. (p = {fmt_pval(freq.z_p_value, loc, 4)})",
        ],
        [
            t("rpu", loc),
            fmt_currency(freq.control_rpu, loc, 2),
            fmt_currency(freq.treatment_rpu, loc, 2),
            f"{fmt_signed_currency(freq.rpu_diff, loc, 2)} (Welch p = {fmt_pval(freq.rpu_welch_p, loc, 4)})",
        ],
        [
            t("aov", loc),
            fmt_currency(freq.control_aov, loc, 2),
            fmt_currency(freq.treatment_aov, loc, 2),
            fmt_signed_currency(freq.treatment_aov - freq.control_aov, loc, 2),
        ],
        [
            t("bayesian_rope_row", loc),
            fmt_pct(bayes.control_post_mean, loc, 2),
            fmt_pct(bayes.treatment_post_mean, loc, 2),
            f"{fmt_pct(bayes.prob_treatment_beats_control, loc, 2)} / {fmt_pct(bayes.prob_above_rope, loc, 2)}",
        ],
        [
            t("projected_30d_title", loc),
            f"{fmt_num(report.config.daily_traffic * 30, loc, 0)} {users_unit}",
            f"{fmt_signed_num(inc_conv, loc, 0)} {conv_unit}",
            f"{fmt_signed_currency(inc_rev, loc, 0)}{mo_unit}",
        ],
    ]
    story.append(
        _styled_table(
            kpi_rows,
            col_widths=[1.9 * inch, 1.3 * inch, 1.3 * inch, 2.7 * inch],
            align_right_cols=(1, 2),
        )
    )
    story.append(Spacer(1, 8))

    # Data Quality & SRM & Power Table (total width = 7.2 in).
    story.append(Paragraph(xml_escape(f"{t('tab_data_quality', loc)} & {t('srm_check', loc)}"), h2_style))
    achieved_word = "Tercapai" if loc == "id" else "Achieved"
    dq_rows = [
        [t("check", loc), t("value", loc), t("details", loc)],
        [
            t("event_cleaning", loc),
            f"{dq.clean_rows:,} / {dq.raw_rows:,} ({fmt_pct(dq.retained_rate, loc, 2)})",
            (
                f"Duplikat: {dq.duplicates_removed}, Campuran: {dq.mixed_group_users_removed}, "
                f"Rev: {dq.invalid_revenue_removed}, Funnel: {dq.invalid_funnel_removed}"
                if loc == "id"
                else f"Dups: {dq.duplicates_removed}, Mixed: {dq.mixed_group_users_removed}, "
                f"Bad Rev: {dq.invalid_revenue_removed}, Funnel: {dq.invalid_funnel_removed}"
            ),
        ],
        [
            t("srm_check", loc),
            t("srm_passed", loc) if srm.passed else t("srm_failed", loc),
            (
                f"{t('control', loc)}={srm.control_n:,}, {t('treatment', loc)}={srm.treatment_n:,}, "
                f"chi2={fmt_num(srm.chi2_stat, loc, 4)}, p={fmt_pval(srm.p_value, loc, 4)}"
            ),
        ],
        [
            t("tab_power", loc),
            f"{achieved_word} {fmt_pct(pwr.achieved_power, loc, 2)}",
            (
                f"Min/varian={pwr.required_sample_per_variant:,}, "
                f"Aktual={pwr.actual_sample_per_variant:,}, Est. {pwr.estimated_duration_days}h"
                if loc == "id"
                else f"Required/arm={pwr.required_sample_per_variant:,}, "
                f"Actual min/arm={pwr.actual_sample_per_variant:,}, Est. {pwr.estimated_duration_days}d"
            ),
        ],
    ]
    story.append(
        _styled_table(dq_rows, col_widths=[2.0 * inch, 1.6 * inch, 3.6 * inch])
    )

    # Prior Sensitivity Table (total width = 7.2 in).
    story.append(Paragraph(xml_escape(t("prior_sensitivity_title", loc)), h2_style))
    sens_rows = [
        [
            t("prior_name", loc),
            t("prob_beats_control", loc),
            t("expected_loss_header", loc),
            t("rel_lift_mean", loc),
            t("hdi_95", loc),
        ],
    ]
    for row in bayes.sensitivity:
        sens_rows.append(
            [
                row.prior_name,
                fmt_pct(row.prob_treatment_beats_control, loc, 2),
                fmt_pct(row.expected_loss_treatment, loc, 4),
                fmt_signed_pct(row.rel_lift_mean, loc, 2),
                f"[{fmt_signed_pct(row.hdi_low, loc, 2)}, {fmt_signed_pct(row.hdi_high, loc, 2)}]",
            ]
        )
    story.append(
        _styled_table(
            sens_rows,
            col_widths=[1.85 * inch, 1.4 * inch, 1.25 * inch, 1.05 * inch, 1.65 * inch],
            align_right_cols=(1, 2, 3),
        )
    )

    # Segment Table (total width = 7.2 in).
    story.append(Paragraph(xml_escape(t("tab_segments", loc)), h2_style))
    seg_rows = [
        [
            t("dimension", loc),
            t("segment", loc),
            t("ctrl_cr", loc),
            t("trt_cr", loc),
            t("relative_lift", loc),
            t("bh_p_value", loc),
            t("significant", loc),
        ],
    ]
    for seg in freq.segments:
        seg_rows.append(
            [
                seg.dimension.title(),
                seg.segment.replace("_", " ").title(),
                fmt_pct(seg.control_cr, loc, 2),
                fmt_pct(seg.treatment_cr, loc, 2),
                fmt_signed_pct(seg.rel_lift, loc, 2),
                fmt_pval(seg.p_value_bh, loc, 4),
                t("yes", loc) if seg.significant_bh else t("no", loc),
            ]
        )
    story.append(
        _styled_table(
            seg_rows,
            col_widths=[0.9 * inch, 1.15 * inch, 0.9 * inch, 0.9 * inch, 0.95 * inch, 1.45 * inch, 0.95 * inch],
            align_right_cols=(2, 3, 4, 5),
        )
    )

    # Figures section on clean subsequent pages.
    figure_order = [
        ("conversion_funnel", t("fig_funnel_title", loc)),
        ("frequentist_ci_segments", t("fig_segments_title", loc)),
        ("bayesian_posteriors", t("fig_bayesian_title", loc)),
        ("power_curve", t("fig_power_title", loc)),
    ]
    existing_figs = [
        (k, title, figure_paths[k])
        for k, title in figure_order
        if k in figure_paths and figure_paths[k].exists()
    ]
    if existing_figs:
        story.append(PageBreak())
        for idx, (_, fig_title, fig_path) in enumerate(existing_figs):
            block = KeepTogether(
                [
                    Paragraph(xml_escape(fig_title), h2_style),
                    _scaled_image(fig_path, max_width=7.1 * inch, max_height=3.6 * inch),
                    Spacer(1, 10),
                ]
            )
            story.append(block)
            if idx == 1 and len(existing_figs) > 2:
                story.append(PageBreak())

    doc.build(story)
    return output_path
