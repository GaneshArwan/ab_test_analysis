"""Synthetic A/B experiment data generation, cleaning, and Pydantic validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from src.schemas import DataQualitySummary, ExperimentConfig, UserEventRecord

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "user_id",
    "timestamp",
    "group",
    "device",
    "channel",
    "viewed_landing",
    "clicked_cta",
    "started_checkout",
    "converted",
    "revenue",
)

_DEVICES: tuple[str, ...] = ("desktop", "mobile", "tablet")
_DEVICE_PROBS: tuple[float, ...] = (0.45, 0.43, 0.12)
_DEVICE_CR_MULT: dict[str, float] = {"desktop": 1.12, "mobile": 0.92, "tablet": 0.84}
_DEVICE_LIFT_MULT: dict[str, float] = {"desktop": 1.35, "mobile": 0.85, "tablet": 0.20}

_CHANNELS: tuple[str, ...] = ("organic", "paid_search", "email", "social")
_CHANNEL_PROBS: tuple[float, ...] = (0.40, 0.30, 0.15, 0.15)
_CHANNEL_CR_MULT: dict[str, float] = {
    "organic": 1.00,
    "paid_search": 0.95,
    "email": 1.25,
    "social": 0.85,
}
_CHANNEL_LIFT_MULT: dict[str, float] = {
    "organic": 0.95,
    "paid_search": 0.90,
    "email": 1.60,
    "social": 0.75,
}

_USER_EVENT_LIST_ADAPTER = TypeAdapter(list[UserEventRecord])


def generate_synthetic_ab_data(
    config: ExperimentConfig,
    output_raw_path: Path | None = None,
) -> pd.DataFrame:
    """Generate deterministic 14-day synthetic A/B user events with ~1.5% realistic anomalies."""
    rng = np.random.default_rng(config.seed)
    n = config.n_users

    user_ids = np.array([f"usr_{i:06d}" for i in range(1, n + 1)])
    start_ts = np.datetime64("2026-10-01T00:00:00")
    offsets_sec = rng.integers(0, 14 * 86400, size=n)
    timestamps = pd.to_datetime(start_ts + offsets_sec.astype("timedelta64[s]"))

    groups = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    devices = rng.choice(_DEVICES, size=n, p=_DEVICE_PROBS)
    channels = rng.choice(_CHANNELS, size=n, p=_CHANNEL_PROBS)

    dev_cr_mult = pd.Series(devices).map(_DEVICE_CR_MULT).to_numpy(dtype=float)
    chan_cr_mult = pd.Series(channels).map(_CHANNEL_CR_MULT).to_numpy(dtype=float)
    dev_lift_mult = pd.Series(devices).map(_DEVICE_LIFT_MULT).to_numpy(dtype=float)
    chan_lift_mult = pd.Series(channels).map(_CHANNEL_LIFT_MULT).to_numpy(dtype=float)

    ctrl_cr = np.clip(config.baseline_cr * dev_cr_mult * chan_cr_mult, 0.002, 0.98)
    is_treatment = groups == "treatment"
    target_cr = np.where(
        is_treatment,
        np.clip(
            ctrl_cr * (1.0 + config.mde_relative * dev_lift_mult * chan_lift_mult),
            0.002,
            0.995,
        ),
        ctrl_cr,
    )

    p_view = np.clip(np.maximum(0.95, target_cr + 0.01), 0.0, 0.996)
    rem = target_cr / p_view
    p_click = np.clip(np.maximum(0.45, np.power(rem, 0.33)), 0.0, 0.998)
    p_checkout = np.clip(
        np.maximum(0.50, np.sqrt(np.clip(rem / p_click, 0.0, 1.0))), 0.0, 0.998
    )
    p_conv_given_checkout = np.clip(
        target_cr / (p_view * p_click * p_checkout), 0.0, 1.0
    )

    viewed_landing = rng.random(n) < p_view
    clicked_cta = viewed_landing & (rng.random(n) < p_click)
    started_checkout = clicked_cta & (rng.random(n) < p_checkout)
    converted = started_checkout & (rng.random(n) < p_conv_given_checkout)

    log_mean = np.where(is_treatment, 4.02, 4.00)
    raw_order_rev = np.round(rng.lognormal(mean=log_mean, sigma=0.5), 2)
    revenue = np.where(converted, raw_order_rev, 0.0)

    clean_base = pd.DataFrame(
        {
            "user_id": user_ids,
            "timestamp": timestamps,
            "group": groups,
            "device": devices,
            "channel": channels,
            "viewed_landing": viewed_landing,
            "clicked_cta": clicked_cta,
            "started_checkout": started_checkout,
            "converted": converted,
            "revenue": revenue,
        }
    )

    # Inject ~1.5% realistic quality anomalies (0.6% duplicates, 0.3% mixed-group,
    # 0.3% invalid revenue, 0.3% broken funnel states).
    n_dups = max(1, round(0.006 * n))
    n_mixed = max(1, round(0.003 * n))
    n_bad_rev = max(1, round(0.003 * n))
    n_bad_funnel = max(1, round(0.003 * n))

    dup_idx = rng.choice(n, size=n_dups, replace=False)
    dups = clean_base.iloc[dup_idx].copy()
    dups["timestamp"] = dups["timestamp"] + pd.to_timedelta(
        rng.integers(5, 3600, size=n_dups), unit="s"
    )

    mixed_idx = rng.choice(n, size=n_mixed, replace=False)
    mixed = clean_base.iloc[mixed_idx].copy()
    mixed["group"] = np.where(mixed["group"] == "control", "treatment", "control")
    mixed["timestamp"] = mixed["timestamp"] + pd.to_timedelta(
        rng.integers(1, 600, size=n_mixed), unit="s"
    )

    bad_rev = pd.DataFrame(
        {
            "user_id": [f"usr_anom_rev_{i:05d}" for i in range(1, n_bad_rev + 1)],
            "timestamp": pd.to_datetime(
                start_ts
                + rng.integers(0, 14 * 86400, size=n_bad_rev).astype("timedelta64[s]")
            ),
            "group": rng.choice(["control", "treatment"], size=n_bad_rev),
            "device": rng.choice(_DEVICES, size=n_bad_rev, p=_DEVICE_PROBS),
            "channel": rng.choice(_CHANNELS, size=n_bad_rev, p=_CHANNEL_PROBS),
            "viewed_landing": True,
            "clicked_cta": True,
            "started_checkout": rng.choice([True, False], size=n_bad_rev),
            "converted": False,
            "revenue": np.where(
                rng.random(n_bad_rev) < 0.35,
                np.nan,
                np.where(rng.random(n_bad_rev) < 0.5, -19.99, 42.50),
            ),
        }
    )

    bad_funnel = pd.DataFrame(
        {
            "user_id": [f"usr_anom_fun_{i:05d}" for i in range(1, n_bad_funnel + 1)],
            "timestamp": pd.to_datetime(
                start_ts
                + rng.integers(0, 14 * 86400, size=n_bad_funnel).astype(
                    "timedelta64[s]"
                )
            ),
            "group": rng.choice(["control", "treatment"], size=n_bad_funnel),
            "device": rng.choice(_DEVICES, size=n_bad_funnel, p=_DEVICE_PROBS),
            "channel": rng.choice(_CHANNELS, size=n_bad_funnel, p=_CHANNEL_PROBS),
            "viewed_landing": False,
            "clicked_cta": True,
            "started_checkout": False,
            "converted": False,
            "revenue": 0.0,
        }
    )

    combined = pd.concat(
        [clean_base, dups, mixed, bad_rev, bad_funnel], ignore_index=True
    )
    combined = combined.iloc[rng.permutation(len(combined))].reset_index(drop=True)

    if output_raw_path is not None:
        output_raw_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_raw_path, index=False)

    return combined


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Parse boolean-like series strictly without treating 'False' or NaN as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    str_mapping: dict[str, bool] = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    def _parse(val: object) -> bool | None:
        if pd.isna(val):
            return None
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, float, np.integer, np.floating)):
            if val == 1:
                return True
            if val == 0:
                return False
            return None
        if isinstance(val, str):
            return str_mapping.get(val.strip().lower())
        return None

    return series.map(_parse).astype("boolean")


def clean_and_validate_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, DataQualitySummary]:
    """Clean raw A/B logs and verify every surviving row against UserEventRecord."""
    missing_cols = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    raw_rows = len(df)
    work = df.loc[:, list(_REQUIRED_COLUMNS)].copy()

    work["user_id"] = work["user_id"].astype("string").str.strip()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce")
    bool_cols = ("viewed_landing", "clicked_cta", "started_checkout", "converted")
    valid_bool_mask = pd.Series(True, index=work.index)
    for col in bool_cols:
        parsed = _coerce_bool_series(work[col])
        valid_bool_mask &= parsed.notna()
        work[col] = parsed.fillna(False).astype(bool)

    # 1. Remove mixed-group users (seen in both arms) and invalid group labels.
    valid_group_mask = work["group"].isin(["control", "treatment"])
    group_nunique = work.groupby("user_id", dropna=False)["group"].transform("nunique")
    mixed_mask = (~valid_group_mask) | (group_nunique > 1) | work["user_id"].isna()
    mixed_group_users_removed = int(mixed_mask.sum())
    work = work.loc[~mixed_mask].copy()
    valid_bool_mask = valid_bool_mask.loc[work.index]

    # 2. Remove invalid revenue rows (NaN, inf, negative, or non-zero on non-converters).
    rev_vals = work["revenue"].to_numpy(dtype=float)
    conv_vals = work["converted"].to_numpy(dtype=bool)
    invalid_rev_mask = (
        ~np.isfinite(rev_vals) | (rev_vals < 0.0) | ((~conv_vals) & (rev_vals != 0.0))
    )
    invalid_revenue_removed = int(invalid_rev_mask.sum())
    work = work.loc[~invalid_rev_mask].copy()
    valid_bool_mask = valid_bool_mask.loc[work.index]

    # 3. Remove invalid funnel rows (unparseable booleans, non-monotonic progression, or missing timestamp).
    v = work["viewed_landing"].to_numpy(dtype=bool)
    c = work["clicked_cta"].to_numpy(dtype=bool)
    s = work["started_checkout"].to_numpy(dtype=bool)
    k = work["converted"].to_numpy(dtype=bool)
    monotonic_ok = (
        valid_bool_mask.to_numpy(dtype=bool)
        & (v >= c)
        & (c >= s)
        & (s >= k)
        & work["timestamp"].notna().to_numpy(dtype=bool)
    )
    invalid_funnel_removed = int((~monotonic_ok).sum())
    work = work.loc[monotonic_ok].copy()

    # 4. Deduplicate user_id keeping earliest timestamp.
    before_dedup = len(work)
    work = (
        work.sort_values(["timestamp", "user_id"], kind="mergesort")
        .drop_duplicates(subset=["user_id"], keep="first")
        .reset_index(drop=True)
    )
    duplicates_removed = before_dedup - len(work)

    # 5. Enforce strict Pydantic v2 schema validation across all surviving records.
    if len(work) > 0:
        _USER_EVENT_LIST_ADAPTER.validate_python(work.to_dict(orient="records"))

    clean_rows = len(work)
    retained_rate = float(clean_rows / raw_rows) if raw_rows > 0 else 0.0

    summary = DataQualitySummary(
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        duplicates_removed=duplicates_removed,
        mixed_group_users_removed=mixed_group_users_removed,
        invalid_revenue_removed=invalid_revenue_removed,
        invalid_funnel_removed=invalid_funnel_removed,
        retained_rate=retained_rate,
    )
    return work, summary


def load_or_generate_data(
    config: ExperimentConfig,
    csv_path: Path = Path("data/ab_test_data.csv"),
) -> tuple[pd.DataFrame, DataQualitySummary]:
    """Generate raw synthetic data, clean and validate it, and persist clean CSV."""
    raw_df = generate_synthetic_ab_data(config)
    clean_df, summary = clean_and_validate_data(raw_df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(csv_path, index=False)
    return clean_df, summary
