"""Strict Pydantic v2 schemas for the A/B experiment analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)


class ExperimentConfig(_StrictSchema):
    n_users: int = Field(default=40000, ge=100, le=5000000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5, allow_inf_nan=False)
    power: float = Field(default=0.80, gt=0.5, lt=0.999, allow_inf_nan=False)
    baseline_cr: float = Field(default=0.10, gt=0.001, lt=0.999, allow_inf_nan=False)
    mde_relative: float = Field(default=0.12, gt=0.005, lt=5.0, allow_inf_nan=False)
    prior_alpha: float = Field(default=10.0, gt=0.0, le=10000.0, allow_inf_nan=False)
    prior_beta: float = Field(default=90.0, gt=0.0, le=10000.0, allow_inf_nan=False)
    rope_margin: float = Field(default=0.005, ge=0.0, lt=0.5, allow_inf_nan=False)
    daily_traffic: int = Field(default=3000, ge=10, le=10000000)

    @model_validator(mode="after")
    def _validate_target_conversion_rate(self) -> ExperimentConfig:
        target_cr = self.baseline_cr * (1.0 + self.mde_relative)
        if target_cr >= 1.0:
            raise ValueError(
                f"Target conversion rate baseline_cr * (1 + mde_relative) = {target_cr:.4f} must be < 1.0."
            )
        return self


class UserEventRecord(_StrictSchema):
    user_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    group: Literal["control", "treatment"]
    device: Literal["desktop", "mobile", "tablet"]
    channel: Literal["organic", "paid_search", "email", "social"]
    viewed_landing: bool
    clicked_cta: bool
    started_checkout: bool
    converted: bool
    revenue: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_funnel_and_revenue(self) -> UserEventRecord:
        if not (
            self.viewed_landing
            >= self.clicked_cta
            >= self.started_checkout
            >= self.converted
        ):
            raise ValueError(
                "Funnel stages must be monotonic: "
                "viewed_landing >= clicked_cta >= started_checkout >= converted."
            )
        if not self.converted and self.revenue != 0.0:
            raise ValueError("Non-converted users must have revenue == 0.0.")
        return self


class DataQualitySummary(_StrictSchema):
    raw_rows: int = Field(ge=0)
    clean_rows: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    mixed_group_users_removed: int = Field(ge=0)
    invalid_revenue_removed: int = Field(ge=0)
    invalid_funnel_removed: int = Field(ge=0)
    retained_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PowerAnalysisResult(_StrictSchema):
    baseline_cr: FiniteFloat
    target_cr: FiniteFloat
    mde_relative: FiniteFloat
    cohens_h: FiniteFloat
    alpha: FiniteFloat
    target_power: FiniteFloat
    required_sample_per_variant: int = Field(ge=1)
    total_required_sample: int = Field(ge=2)
    estimated_duration_days: int = Field(ge=1)
    actual_sample_per_variant: int = Field(ge=0)
    achieved_power: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    power_curve: list[dict[str, FiniteFloat]]


class SRMResult(_StrictSchema):
    control_n: int = Field(ge=0)
    treatment_n: int = Field(ge=0)
    expected_ratio: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    chi2_stat: float = Field(ge=0.0, allow_inf_nan=False)
    p_value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    passed: bool


class FunnelStageResult(_StrictSchema):
    stage: str
    control_n: int = Field(ge=0)
    control_rate: FiniteFloat
    treatment_n: int = Field(ge=0)
    treatment_rate: FiniteFloat
    relative_lift: FiniteFloat
    p_value: FiniteFloat


class SegmentResult(_StrictSchema):
    dimension: str
    segment: str
    control_n: int = Field(ge=0)
    control_cr: FiniteFloat
    treatment_n: int = Field(ge=0)
    treatment_cr: FiniteFloat
    abs_diff: FiniteFloat
    rel_lift: FiniteFloat
    ci_low: FiniteFloat
    ci_high: FiniteFloat
    p_value: FiniteFloat
    p_value_bh: FiniteFloat
    significant_bh: bool


class FrequentistResult(_StrictSchema):
    control_n: int = Field(ge=0)
    control_conversions: int = Field(ge=0)
    control_cr: FiniteFloat
    control_ci_low: FiniteFloat
    control_ci_high: FiniteFloat
    treatment_n: int = Field(ge=0)
    treatment_conversions: int = Field(ge=0)
    treatment_cr: FiniteFloat
    treatment_ci_low: FiniteFloat
    treatment_ci_high: FiniteFloat
    abs_diff: FiniteFloat
    rel_lift: FiniteFloat
    diff_ci_low: FiniteFloat
    diff_ci_high: FiniteFloat
    z_stat: FiniteFloat
    z_p_value: FiniteFloat
    chi2_stat: FiniteFloat
    chi2_p_value: FiniteFloat
    cohens_h: FiniteFloat
    significant: bool
    control_rpu: FiniteFloat
    treatment_rpu: FiniteFloat
    rpu_diff: FiniteFloat
    rpu_ci_low: FiniteFloat
    rpu_ci_high: FiniteFloat
    rpu_welch_t: FiniteFloat
    rpu_welch_p: FiniteFloat
    rpu_mannwhitney_u: FiniteFloat
    rpu_mannwhitney_p: FiniteFloat
    rpu_cohens_d: FiniteFloat
    control_aov: FiniteFloat
    treatment_aov: FiniteFloat
    funnel_stages: list[FunnelStageResult]
    segments: list[SegmentResult]


class PriorSensitivityResult(_StrictSchema):
    prior_name: str
    prior_alpha: float = Field(gt=0.0, allow_inf_nan=False)
    prior_beta: float = Field(gt=0.0, allow_inf_nan=False)
    prob_treatment_beats_control: FiniteFloat
    expected_loss_treatment: FiniteFloat
    rel_lift_mean: FiniteFloat
    hdi_low: FiniteFloat
    hdi_high: FiniteFloat


class BayesianResult(_StrictSchema):
    prior_alpha: float = Field(gt=0.0, allow_inf_nan=False)
    prior_beta: float = Field(gt=0.0, allow_inf_nan=False)
    control_post_alpha: FiniteFloat
    control_post_beta: FiniteFloat
    treatment_post_alpha: FiniteFloat
    treatment_post_beta: FiniteFloat
    control_post_mean: FiniteFloat
    treatment_post_mean: FiniteFloat
    prob_treatment_beats_control: FiniteFloat
    expected_loss_treatment: FiniteFloat
    expected_loss_control: FiniteFloat
    abs_diff_mean: FiniteFloat
    abs_diff_hdi_low: FiniteFloat
    abs_diff_hdi_high: FiniteFloat
    rel_lift_mean: FiniteFloat
    rel_lift_hdi_low: FiniteFloat
    rel_lift_hdi_high: FiniteFloat
    rope_margin: FiniteFloat
    prob_in_rope: FiniteFloat
    prob_above_rope: FiniteFloat
    prob_below_rope: FiniteFloat
    sensitivity: list[PriorSensitivityResult]


class AnalysisReport(_StrictSchema):
    generated_at: datetime
    locale: Literal["en", "id"]
    config: ExperimentConfig
    data_quality: DataQualitySummary
    power_analysis: PowerAnalysisResult
    srm: SRMResult
    frequentist: FrequentistResult
    bayesian: BayesianResult
    decision: Literal["SHIP", "HOLD", "ITERATE"]
    recommendation_reason: str
