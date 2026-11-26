"""Unit tests for Pydantic schemas (`src/schemas.py`) and i18n utilities (`src/i18n.py`)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src import i18n
from src.i18n import SUPPORTED_LOCALES, fmt_currency, fmt_num, fmt_pct, t
from src.schemas import ExperimentConfig, UserEventRecord


@pytest.fixture
def valid_event_payload() -> dict[str, object]:
    """Return a baseline valid converted UserEventRecord payload."""
    return {
        "user_id": "usr_000001",
        "timestamp": datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC),
        "group": "treatment",
        "device": "desktop",
        "channel": "organic",
        "viewed_landing": True,
        "clicked_cta": True,
        "started_checkout": True,
        "converted": True,
        "revenue": 54.25,
    }


def test_experiment_config_defaults_are_valid() -> None:
    # Arrange & Act
    cfg = ExperimentConfig()

    # Assert
    assert cfg.n_users == 40000
    assert cfg.seed == 42
    assert cfg.alpha == pytest.approx(0.05)
    assert cfg.power == pytest.approx(0.80)
    assert cfg.baseline_cr == pytest.approx(0.10)
    assert cfg.mde_relative == pytest.approx(0.12)
    assert cfg.prior_alpha == pytest.approx(10.0)
    assert cfg.prior_beta == pytest.approx(90.0)
    assert cfg.rope_margin == pytest.approx(0.005)
    assert cfg.daily_traffic == 3000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_users", 100),
        ("n_users", 5000000),
        ("seed", 0),
        ("seed", 2**31 - 1),
        ("alpha", 0.001),
        ("alpha", 0.499),
        ("power", 0.501),
        ("power", 0.998),
        ("baseline_cr", 0.002),
        ("baseline_cr", 0.85),
        ("mde_relative", 0.006),
        ("mde_relative", 4.99),
        ("prior_alpha", 0.5),
        ("prior_alpha", 10000.0),
        ("prior_beta", 0.5),
        ("prior_beta", 10000.0),
        ("rope_margin", 0.0),
        ("rope_margin", 0.49),
        ("daily_traffic", 10),
        ("daily_traffic", 10000000),
    ],
)
def test_experiment_config_valid_boundaries_succeed(field: str, value: float) -> None:
    # Arrange
    kwargs = {field: value}

    # Act
    cfg = ExperimentConfig(**kwargs)

    # Assert
    assert getattr(cfg, field) == value


def test_experiment_config_extra_field_raises_validation_error() -> None:
    # Arrange
    payload = {"unknown_param": 123}

    # Act & Assert
    with pytest.raises(ValidationError):
        ExperimentConfig(**payload)


def test_experiment_config_target_cr_ge_one_raises_validation_error() -> None:
    # Arrange / Act & Assert
    with pytest.raises(ValidationError, match="Target conversion rate"):
        ExperimentConfig(baseline_cr=0.50, mde_relative=1.00)


@pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [
        "alpha",
        "power",
        "baseline_cr",
        "mde_relative",
        "prior_alpha",
        "prior_beta",
        "rope_margin",
    ],
)
def test_experiment_config_non_finite_float_raises_validation_error(
    field: str, bad_float: float
) -> None:
    # Arrange
    payload = {field: bad_float}

    # Act & Assert
    with pytest.raises(ValidationError):
        ExperimentConfig(**payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n_users", 99),
        ("n_users", 5000001),
        ("seed", -1),
        ("alpha", 0.0),
        ("alpha", -0.05),
        ("alpha", 0.5),
        ("power", 0.5),
        ("power", 0.999),
        ("baseline_cr", 0.001),
        ("baseline_cr", 0.999),
        ("mde_relative", 0.005),
        ("mde_relative", 5.0),
        ("prior_alpha", 0.0),
        ("prior_beta", -1.0),
        ("rope_margin", -0.01),
        ("rope_margin", 0.5),
        ("daily_traffic", 9),
    ],
)
def test_experiment_config_out_of_range_raises_validation_error(
    field: str, bad_value: float
) -> None:
    # Arrange
    payload = {field: bad_value}

    # Act & Assert
    with pytest.raises(ValidationError):
        ExperimentConfig(**payload)


def test_user_event_record_valid_converted_and_non_converted_succeed(
    valid_event_payload: dict[str, object],
) -> None:
    # Arrange
    non_conv_payload = {
        **valid_event_payload,
        "group": "control",
        "converted": False,
        "revenue": 0.0,
    }

    # Act
    conv_rec = UserEventRecord(**valid_event_payload)
    non_conv_rec = UserEventRecord(**non_conv_payload)

    # Assert
    assert conv_rec.converted is True
    assert conv_rec.revenue == pytest.approx(54.25)
    assert non_conv_rec.converted is False
    assert non_conv_rec.revenue == pytest.approx(0.0)


def test_user_event_record_extra_field_raises_validation_error(
    valid_event_payload: dict[str, object],
) -> None:
    # Arrange
    payload = {**valid_event_payload, "extra_col": "nope"}

    # Act & Assert
    with pytest.raises(ValidationError):
        UserEventRecord(**payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("group", "variant_b"),
        ("device", "smartwatch"),
        ("channel", "referral"),
        ("user_id", ""),
    ],
)
def test_user_event_record_invalid_categorical_or_id_raises_validation_error(
    valid_event_payload: dict[str, object], field: str, bad_value: str
) -> None:
    # Arrange
    payload = {**valid_event_payload, field: bad_value}

    # Act & Assert
    with pytest.raises(ValidationError):
        UserEventRecord(**payload)


@pytest.mark.parametrize(
    "bad_revenue",
    [-0.01, -100.0, float("nan"), float("inf"), float("-inf")],
)
def test_user_event_record_invalid_revenue_raises_validation_error(
    valid_event_payload: dict[str, object], bad_revenue: float
) -> None:
    # Arrange
    payload = {**valid_event_payload, "revenue": bad_revenue}

    # Act & Assert
    with pytest.raises(ValidationError):
        UserEventRecord(**payload)


@pytest.mark.parametrize(
    ("viewed", "clicked", "started", "converted"),
    [
        (False, True, False, False),
        (False, True, True, True),
        (True, False, True, False),
        (True, True, False, True),
        (False, False, False, True),
    ],
)
def test_user_event_record_non_monotonic_funnel_raises_validation_error(
    valid_event_payload: dict[str, object],
    viewed: bool,
    clicked: bool,
    started: bool,
    converted: bool,
) -> None:
    # Arrange
    payload = {
        **valid_event_payload,
        "viewed_landing": viewed,
        "clicked_cta": clicked,
        "started_checkout": started,
        "converted": converted,
        "revenue": 25.0 if converted else 0.0,
    }

    # Act & Assert
    with pytest.raises(ValidationError, match="Funnel stages must be monotonic"):
        UserEventRecord(**payload)


def test_user_event_record_positive_revenue_when_not_converted_raises_validation_error(
    valid_event_payload: dict[str, object],
) -> None:
    # Arrange
    payload = {**valid_event_payload, "converted": False, "revenue": 19.99}

    # Act & Assert
    with pytest.raises(ValidationError, match="Non-converted users must have revenue == 0.0"):
        UserEventRecord(**payload)


@pytest.mark.parametrize(
    ("locale", "key", "expected"),
    [
        ("en", "decision_SHIP", "SHIP"),
        ("en", "decision_HOLD", "HOLD"),
        ("en", "decision_ITERATE", "ITERATE"),
        ("id", "decision_SHIP", "RILIS"),
        ("id", "decision_HOLD", "TAHAN"),
        ("id", "decision_ITERATE", "ITERASI"),
        ("en", "srm_passed", "PASSED"),
        ("id", "srm_passed", "LOLOS"),
        ("en", "srm_failed", "FAILED"),
        ("id", "srm_failed", "GAGAL"),
    ],
)
def test_t_decision_and_status_keys_return_expected_translations(
    locale: str, key: str, expected: str
) -> None:
    # Arrange & Act
    result = t(key, locale=locale)

    # Assert
    assert result == expected


def test_t_all_keys_match_between_supported_locales() -> None:
    # Arrange
    en_keys = set(i18n._TRANSLATIONS["en"].keys())

    # Act & Assert
    for locale in SUPPORTED_LOCALES:
        assert set(i18n._TRANSLATIONS[locale].keys()) == en_keys


def test_t_formats_template_with_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setitem(i18n._TRANSLATIONS["en"], "_test_fmt", "Users: {count}, Arm: {arm}")

    # Act
    formatted = t("_test_fmt", locale="en", count=150, arm="treatment")

    # Assert
    assert formatted == "Users: 150, Arm: treatment"


def test_fmt_num_fmt_pct_fmt_currency_format_en_and_id_correctly() -> None:
    # Arrange
    number = 12345.678
    prop = 0.1125
    amount = 1250.5

    # Act
    en_num = fmt_num(number, locale="en", decimals=2)
    id_num = fmt_num(number, locale="id", decimals=2)
    en_pct = fmt_pct(prop, locale="en", decimals=2)
    id_pct = fmt_pct(prop, locale="id", decimals=1)
    en_cur = fmt_currency(amount, locale="en", decimals=2)
    id_cur = fmt_currency(amount, locale="id", decimals=0)

    # Assert
    assert en_num == "12,345.68"
    assert id_num == "12.345,68"
    assert en_pct == "11.25%"
    assert id_pct == "11,2%" or id_pct == "11,3%"  # Babel half-even rounding
    assert en_cur == "$1,250.50"
    assert "1.250" in id_cur or "1.251" in id_cur


@pytest.mark.parametrize("bad_locale", ["fr", "de", "ja", ""])
def test_i18n_functions_raise_value_error_on_unsupported_locale(bad_locale: str) -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Unsupported locale"):
        t("app_title", locale=bad_locale)
    with pytest.raises(ValueError, match="Unsupported locale"):
        fmt_num(10.0, locale=bad_locale)
    with pytest.raises(ValueError, match="Unsupported locale"):
        fmt_pct(0.1, locale=bad_locale)
    with pytest.raises(ValueError, match="Unsupported locale"):
        fmt_currency(10.0, locale=bad_locale)


def test_fmt_num_negative_decimals_raises_value_error() -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        fmt_num(10.0, locale="en", decimals=-1)


def test_fmt_signed_and_pval_handle_positive_negative_and_tiny_values() -> None:
    # Arrange & Act & Assert
    assert i18n.fmt_signed_num(1.25, "en", 2) == "+1.25"
    assert i18n.fmt_signed_num(-1.25, "en", 2) == "-1.25"
    assert i18n.fmt_signed_pct(0.05, "en", 1) == "+5.0%"
    assert i18n.fmt_signed_pct(-0.05, "en", 1) == "-5.0%"
    assert i18n.fmt_signed_currency(2.5, "en", 2) == "+$2.50"
    assert i18n.fmt_signed_currency(-2.5, "en", 2) == "-$2.50"
    assert fmt_currency(109602.71, "en", 0) == "$109,603"
    assert i18n.fmt_signed_currency(109602.71, "en", 0) == "+$109,603"
    assert i18n.fmt_pval(1e-7, "en", 4) == "< 0.0001"
    assert i18n.fmt_pval(1e-7, "id", 4) == "< 0,0001"
    assert i18n.fmt_pval(0.042, "en", 4) == "0.0420"
