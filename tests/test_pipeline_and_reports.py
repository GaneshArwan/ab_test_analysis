"""Integration, report decision branch, CLI, and app module smoke tests."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import app
from src.bayesian import run_bayesian_analysis
from src.data_pipeline import load_or_generate_data
from src.frequentist import check_srm, run_frequentist_analysis
from src.power import run_power_analysis
from src.report import build_analysis_report, write_executive_summary_md
from src.run_analysis import main, run_pipeline
from src.schemas import AnalysisReport, ExperimentConfig


def _build_base_components(
    tmp_path: Path, n_users: int = 1200, seed: int = 42
) -> tuple[ExperimentConfig, dict[str, object]]:
    cfg = ExperimentConfig(n_users=n_users, seed=seed)
    clean_df, dq = load_or_generate_data(cfg, csv_path=tmp_path / "base_ab.csv")
    ctrl_n = int((clean_df["group"] == "control").sum())
    trt_n = int((clean_df["group"] == "treatment").sum())
    pwr = run_power_analysis(cfg, actual_control_n=ctrl_n, actual_treatment_n=trt_n)
    srm = check_srm(control_n=ctrl_n, treatment_n=trt_n)
    freq = run_frequentist_analysis(clean_df, alpha=cfg.alpha)
    bayes = run_bayesian_analysis(
        control_n=ctrl_n,
        control_conv=freq.control_conversions,
        treatment_n=trt_n,
        treatment_conv=freq.treatment_conversions,
        config=cfg,
        n_draws=5000,
    )
    return cfg, {
        "data_quality": dq,
        "power_analysis": pwr,
        "srm": srm,
        "frequentist": freq,
        "bayesian": bayes,
    }


@pytest.mark.parametrize("locale", ["en", "id"])
def test_run_pipeline_end_to_end_creates_all_artifacts(
    locale: str, tmp_path: Path
) -> None:
    # Arrange
    cfg = ExperimentConfig(n_users=1500, seed=42)
    out_dir = tmp_path / f"reports_{locale}"
    data_path = tmp_path / f"data_{locale}" / "ab_test_data.csv"

    # Act
    report, figure_paths = run_pipeline(
        config=cfg,
        locale=locale,
        output_dir=out_dir,
        data_path=data_path,
    )

    # Assert
    json_path = out_dir / "analysis_results.json"
    md_path = out_dir / "executive_summary.md"
    pdf_path = out_dir / "ab_test_report.pdf"

    assert data_path.exists()
    assert json_path.exists()
    parsed = AnalysisReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert parsed.locale == locale

    assert len(figure_paths) == 4
    for fig_path in figure_paths.values():
        assert fig_path.exists()
        raw_bytes = fig_path.read_bytes()
        assert len(raw_bytes) > 0
        assert raw_bytes.startswith(b"\x89PNG")

    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert report.decision in md_text
    for placeholder in ("{{", "}}", "[TODO", "[INSERT", "[PLACEHOLDER", "<TODO"):
        assert placeholder not in md_text

    assert pdf_path.exists()
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")


def test_run_pipeline_invalid_locale_raises_value_error(tmp_path: Path) -> None:
    # Arrange
    cfg = ExperimentConfig(n_users=500)

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported locale"):
        run_pipeline(
            config=cfg,
            locale="fr",
            output_dir=tmp_path / "reports",
            data_path=tmp_path / "ab.csv",
        )


@pytest.mark.parametrize("locale", ["en", "id"])
def test_build_analysis_report_decision_branches(
    locale: str, tmp_path: Path
) -> None:
    # Arrange
    cfg, parts = _build_base_components(tmp_path)
    dq = parts["data_quality"]
    pwr = parts["power_analysis"]
    srm_pass = parts["srm"].model_copy(update={"passed": True, "p_value": 0.85})
    srm_fail = parts["srm"].model_copy(update={"passed": False, "p_value": 0.0001})
    freq_sig = parts["frequentist"].model_copy(
        update={
            "significant": True,
            "z_p_value": 0.001,
            "segments": [
                s.model_copy(update={"significant_bh": False})
                for s in parts["frequentist"].segments
            ],
        }
    )
    freq_nonsig = parts["frequentist"].model_copy(
        update={"significant": False, "z_p_value": 0.35}
    )
    bayes_strong = parts["bayesian"].model_copy(
        update={"prob_treatment_beats_control": 0.995, "prob_above_rope": 0.94}
    )
    bayes_weak = parts["bayesian"].model_copy(
        update={"prob_treatment_beats_control": 0.72, "prob_above_rope": 0.40}
    )
    fixed_ts = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)

    # Act
    hold_report = build_analysis_report(
        cfg, dq, pwr, srm_fail, freq_sig, bayes_strong, locale=locale, generated_at=fixed_ts
    )
    ship_report = build_analysis_report(
        cfg, dq, pwr, srm_pass, freq_sig, bayes_strong, locale=locale, generated_at=fixed_ts
    )
    iterate_report = build_analysis_report(
        cfg, dq, pwr, srm_pass, freq_nonsig, bayes_weak, locale=locale, generated_at=fixed_ts
    )
    md_out = write_executive_summary_md(
        ship_report, tmp_path / f"summary_nosig_{locale}.md"
    )

    # Assert
    assert hold_report.decision == "HOLD"
    assert ship_report.decision == "SHIP"
    assert iterate_report.decision == "ITERATE"
    assert md_out.exists()
    assert len(md_out.read_text(encoding="utf-8")) > 100


def test_build_analysis_report_invalid_locale_raises_value_error(
    tmp_path: Path,
) -> None:
    # Arrange
    cfg, parts = _build_base_components(tmp_path)

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported locale"):
        build_analysis_report(
            cfg,
            parts["data_quality"],
            parts["power_analysis"],
            parts["srm"],
            parts["frequentist"],
            parts["bayesian"],
            locale="es",
        )


def test_cli_main_valid_and_invalid_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # Arrange
    out_dir = tmp_path / "cli_reports"
    data_path = tmp_path / "cli_data" / "ab.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_analysis",
            "--locale",
            "en",
            "--seed",
            "42",
            "--n-users",
            "1000",
            "--output-dir",
            str(out_dir),
            "--data-path",
            str(data_path),
        ],
    )

    # Act
    exit_ok = main()
    captured_ok = capsys.readouterr()

    # Assert
    assert exit_ok == 0
    assert "Analysis complete" in captured_ok.out
    assert (out_dir / "analysis_results.json").exists()
    assert (out_dir / "ab_test_report.pdf").exists()

    # Arrange invalid n_users
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_analysis", "--n-users", "10"],
    )

    # Act
    exit_bad = main()
    captured_bad = capsys.readouterr()

    # Assert
    assert exit_bad == 2
    assert "Configuration validation error" in captured_bad.out


def test_app_module_exposes_callable_main_and_decision_badge(tmp_path: Path) -> None:
    # Arrange
    cfg, parts = _build_base_components(tmp_path, n_users=1000, seed=42)
    report = build_analysis_report(
        cfg,
        parts["data_quality"],
        parts["power_analysis"],
        parts["srm"],
        parts["frequentist"],
        parts["bayesian"],
        locale="en",
    )

    # Act
    badge_html = app._decision_badge_html(report, locale="en")

    # Assert
    assert callable(app.main)
    assert report.decision in badge_html
    assert 'role="status"' in badge_html
