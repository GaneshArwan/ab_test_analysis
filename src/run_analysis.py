"""Reproducible CLI and programmatic pipeline runner for A/B experiment analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from src.bayesian import run_bayesian_analysis
from src.data_pipeline import load_or_generate_data
from src.frequentist import check_srm, run_frequentist_analysis
from src.i18n import SUPPORTED_LOCALES, validate_locale
from src.power import run_power_analysis
from src.report import (
    build_analysis_report,
    write_executive_summary_md,
    write_pdf_report,
)
from src.schemas import AnalysisReport, ExperimentConfig
from src.visualizations import generate_all_figures


def run_pipeline(
    config: ExperimentConfig | None = None,
    locale: str = "en",
    output_dir: Path = Path("reports"),
    data_path: Path = Path("data/ab_test_data.csv"),
) -> tuple[AnalysisReport, dict[str, Path]]:
    """Run the end-to-end A/B test analysis pipeline and persist all outputs."""
    loc = validate_locale(locale)

    cfg = config or ExperimentConfig()
    clean_df, data_quality = load_or_generate_data(cfg, csv_path=data_path)

    ctrl_n = int((clean_df["group"] == "control").sum())
    trt_n = int((clean_df["group"] == "treatment").sum())

    power_analysis = run_power_analysis(
        cfg, actual_control_n=ctrl_n, actual_treatment_n=trt_n
    )
    srm = check_srm(control_n=ctrl_n, treatment_n=trt_n)
    frequentist = run_frequentist_analysis(clean_df, alpha=cfg.alpha)
    bayesian = run_bayesian_analysis(
        control_n=ctrl_n,
        control_conv=frequentist.control_conversions,
        treatment_n=trt_n,
        treatment_conv=frequentist.treatment_conversions,
        config=cfg,
    )

    report = build_analysis_report(
        config=cfg,
        data_quality=data_quality,
        power_analysis=power_analysis,
        srm=srm,
        frequentist=frequentist,
        bayesian=bayesian,
        locale=loc,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_all_figures(
        frequentist=frequentist,
        bayesian=bayesian,
        power=power_analysis,
        output_dir=output_dir / "figures",
        locale=loc,
        seed=cfg.seed,
    )
    write_executive_summary_md(report, output_dir / "executive_summary.md")
    write_pdf_report(report, figure_paths, output_dir / "ab_test_report.pdf")
    (output_dir / "analysis_results.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report, figure_paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the A/B experiment statistical analysis pipeline."
    )
    parser.add_argument(
        "--locale",
        choices=SUPPORTED_LOCALES,
        default="en",
        help="Output locale for reports and figures (en or id).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation and Bayesian draws.",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=40000,
        help="Total synthetic users to generate before cleaning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for JSON, Markdown, PDF, and figure outputs.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ab_test_data.csv"),
        help="Path to save the cleaned experiment dataset CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = ExperimentConfig(seed=args.seed, n_users=args.n_users)
    except ValidationError as exc:
        print(f"Configuration validation error: {exc}")
        return 2

    report, figure_paths = run_pipeline(
        config=config,
        locale=args.locale,
        output_dir=args.output_dir,
        data_path=args.data_path,
    )
    print(
        f"Analysis complete | Decision={report.decision} | "
        f"SRM={'PASS' if report.srm.passed else 'FAIL'} | "
        f"Control CR={report.frequentist.control_cr:.4f} | "
        f"Treatment CR={report.frequentist.treatment_cr:.4f} | "
        f"Rel Lift={report.frequentist.rel_lift:.2%} | "
        f"p={report.frequentist.z_p_value:.4f} | "
        f"P(T>C)={report.bayesian.prob_treatment_beats_control:.4f} | "
        f"Figures={len(figure_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
