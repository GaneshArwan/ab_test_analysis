"""Unit tests for synthetic data generation, cleaning, and schema validation (`src/data_pipeline.py`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from src.data_pipeline import (
    clean_and_validate_data,
    generate_synthetic_ab_data,
    load_or_generate_data,
)
from src.schemas import ExperimentConfig, UserEventRecord

_RECORD_LIST_ADAPTER = TypeAdapter(list[UserEventRecord])


@pytest.fixture
def small_config() -> ExperimentConfig:
    """Return a fast deterministic ExperimentConfig for pipeline unit tests."""
    return ExperimentConfig(n_users=1000, seed=123)


def _base_valid_rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": "u_clean_1",
            "timestamp": "2026-10-01T10:00:00",
            "group": "control",
            "device": "desktop",
            "channel": "organic",
            "viewed_landing": True,
            "clicked_cta": True,
            "started_checkout": True,
            "converted": True,
            "revenue": 50.0,
        },
        {
            "user_id": "u_clean_2",
            "timestamp": "2026-10-01T11:00:00",
            "group": "treatment",
            "device": "mobile",
            "channel": "email",
            "viewed_landing": True,
            "clicked_cta": False,
            "started_checkout": False,
            "converted": False,
            "revenue": 0.0,
        },
    ]


def test_generate_and_clean_identical_seed_produces_identical_outputs(
    small_config: ExperimentConfig,
) -> None:
    # Arrange
    cfg_a = small_config
    cfg_b = ExperimentConfig(n_users=1000, seed=123)

    # Act
    raw_a = generate_synthetic_ab_data(cfg_a)
    raw_b = generate_synthetic_ab_data(cfg_b)
    clean_a, summary_a = clean_and_validate_data(raw_a)
    clean_b, summary_b = clean_and_validate_data(raw_b)

    # Assert
    pd.testing.assert_frame_equal(raw_a, raw_b)
    pd.testing.assert_frame_equal(clean_a, clean_b)
    assert summary_a == summary_b


def test_generate_synthetic_ab_data_different_seed_produces_different_data() -> None:
    # Arrange
    cfg_1 = ExperimentConfig(n_users=500, seed=11)
    cfg_2 = ExperimentConfig(n_users=500, seed=22)

    # Act
    raw_1 = generate_synthetic_ab_data(cfg_1)
    raw_2 = generate_synthetic_ab_data(cfg_2)

    # Assert
    assert not raw_1["revenue"].equals(raw_2["revenue"])


def test_generate_synthetic_ab_data_writes_raw_csv_when_path_provided(
    small_config: ExperimentConfig, tmp_path: Path
) -> None:
    # Arrange
    raw_path = tmp_path / "raw" / "events.csv"

    # Act
    raw_df = generate_synthetic_ab_data(small_config, output_raw_path=raw_path)

    # Assert
    assert raw_path.exists()
    loaded = pd.read_csv(raw_path)
    assert len(loaded) == len(raw_df)


def test_clean_and_validate_data_removes_duplicates_keeping_earliest_timestamp() -> None:
    # Arrange
    rows = _base_valid_rows()
    rows.append(
        {
            **rows[0],
            "timestamp": "2026-10-01T15:30:00",
            "revenue": 999.0,
        }
    )
    rows.append(
        {
            **rows[0],
            "timestamp": "2026-10-01T08:15:00",
            "revenue": 75.0,
        }
    )
    df = pd.DataFrame(rows)

    # Act
    clean_df, summary = clean_and_validate_data(df)

    # Assert
    assert summary.raw_rows == 4
    assert summary.duplicates_removed == 2
    assert summary.clean_rows == 2
    surviving = clean_df.loc[clean_df["user_id"] == "u_clean_1"].iloc[0]
    assert surviving["revenue"] == pytest.approx(75.0)
    assert pd.Timestamp(surviving["timestamp"]) == pd.Timestamp("2026-10-01T08:15:00")


def test_clean_and_validate_data_drops_mixed_group_users_completely() -> None:
    # Arrange
    rows = _base_valid_rows()
    rows.extend(
        [
            {
                **rows[0],
                "user_id": "u_mixed",
                "group": "control",
                "timestamp": "2026-10-01T09:00:00",
            },
            {
                **rows[0],
                "user_id": "u_mixed",
                "group": "treatment",
                "timestamp": "2026-10-01T09:05:00",
            },
        ]
    )
    df = pd.DataFrame(rows)

    # Act
    clean_df, summary = clean_and_validate_data(df)

    # Assert
    assert "u_mixed" not in set(clean_df["user_id"])
    assert summary.mixed_group_users_removed == 2
    assert summary.clean_rows == 2


def test_clean_and_validate_data_parses_string_booleans_and_drops_invalid_bools() -> None:
    # Arrange
    base = _base_valid_rows()[0]
    df = pd.DataFrame(
        [
            {
                **base,
                "user_id": "u_str_false",
                "viewed_landing": "true",
                "clicked_cta": "False",
                "started_checkout": "0",
                "converted": "false",
                "revenue": 0.0,
            },
            {
                **base,
                "user_id": "u_bad_bool",
                "viewed_landing": "maybe",
                "clicked_cta": False,
                "started_checkout": False,
                "converted": False,
                "revenue": 0.0,
            },
        ]
    )

    # Act
    clean_df, summary = clean_and_validate_data(df)

    # Assert
    assert summary.clean_rows == 1
    assert summary.invalid_funnel_removed == 1
    row = clean_df.iloc[0]
    assert row["user_id"] == "u_str_false"
    assert bool(row["viewed_landing"]) is True
    assert bool(row["clicked_cta"]) is False
    assert bool(row["converted"]) is False


def test_clean_and_validate_data_drops_invalid_revenues_and_broken_funnels() -> None:
    # Arrange
    base = _base_valid_rows()[0]
    bad_rows = [
        {**base, "user_id": "u_neg_rev", "revenue": -5.0},
        {**base, "user_id": "u_nan_rev", "revenue": np.nan},
        {**base, "user_id": "u_inf_rev", "revenue": np.inf},
        {**base, "user_id": "u_nonconv_rev", "converted": False, "revenue": 12.0},
        {
            **base,
            "user_id": "u_bad_funnel_1",
            "viewed_landing": False,
            "clicked_cta": True,
            "started_checkout": False,
            "converted": False,
            "revenue": 0.0,
        },
        {
            **base,
            "user_id": "u_bad_funnel_2",
            "viewed_landing": True,
            "clicked_cta": True,
            "started_checkout": False,
            "converted": True,
            "revenue": 10.0,
        },
        {
            **base,
            "user_id": "u_bad_ts",
            "timestamp": "not-a-timestamp",
        },
    ]
    df = pd.DataFrame(_base_valid_rows() + bad_rows)

    # Act
    clean_df, summary = clean_and_validate_data(df)

    # Assert
    assert summary.invalid_revenue_removed == 4
    assert summary.invalid_funnel_removed == 3
    assert summary.clean_rows == 2
    assert set(clean_df["user_id"]) == {"u_clean_1", "u_clean_2"}


def test_clean_and_validate_data_all_surviving_rows_pass_user_event_record(
    small_config: ExperimentConfig,
) -> None:
    # Arrange
    raw_df = generate_synthetic_ab_data(small_config)

    # Act
    clean_df, summary = clean_and_validate_data(raw_df)
    validated = _RECORD_LIST_ADAPTER.validate_python(clean_df.to_dict(orient="records"))

    # Assert
    assert len(validated) == summary.clean_rows
    assert summary.duplicates_removed >= 1
    assert summary.mixed_group_users_removed >= 1
    assert summary.invalid_revenue_removed >= 1
    assert summary.invalid_funnel_removed >= 1
    assert 0.90 < summary.retained_rate < 1.0


def test_clean_and_validate_data_missing_columns_raises_value_error() -> None:
    # Arrange
    df = pd.DataFrame({"user_id": ["u1"], "group": ["control"]})

    # Act & Assert
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_and_validate_data(df)


def test_clean_and_validate_data_empty_dataframe_returns_zero_summary() -> None:
    # Arrange
    df = pd.DataFrame(_base_valid_rows()).iloc[0:0]

    # Act
    clean_df, summary = clean_and_validate_data(df)

    # Assert
    assert len(clean_df) == 0
    assert summary.raw_rows == 0
    assert summary.clean_rows == 0
    assert summary.retained_rate == pytest.approx(0.0)


def test_load_or_generate_data_persists_clean_csv(
    small_config: ExperimentConfig, tmp_path: Path
) -> None:
    # Arrange
    csv_path = tmp_path / "nested" / "ab.csv"

    # Act
    clean_df, summary = load_or_generate_data(small_config, csv_path=csv_path)

    # Assert
    assert csv_path.exists()
    persisted = pd.read_csv(csv_path)
    assert len(persisted) == len(clean_df) == summary.clean_rows
    assert list(persisted.columns) == list(clean_df.columns)
