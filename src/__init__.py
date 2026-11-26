"""Core statistical analysis package for A/B experiment evaluation."""

from src.bayesian import compute_hdi, run_bayesian_analysis
from src.data_pipeline import (
    clean_and_validate_data,
    generate_synthetic_ab_data,
    load_or_generate_data,
)
from src.frequentist import (
    check_srm,
    newcombe_wilson_diff_ci,
    run_frequentist_analysis,
    wilson_ci,
)
from src.i18n import SUPPORTED_LOCALES, fmt_currency, fmt_num, fmt_pct, t
from src.power import calculate_sample_size, compute_cohens_h, run_power_analysis
from src.schemas import (
    AnalysisReport,
    BayesianResult,
    DataQualitySummary,
    ExperimentConfig,
    FrequentistResult,
    FunnelStageResult,
    PowerAnalysisResult,
    PriorSensitivityResult,
    SegmentResult,
    SRMResult,
    UserEventRecord,
)

__all__ = [
    "SUPPORTED_LOCALES",
    "AnalysisReport",
    "BayesianResult",
    "DataQualitySummary",
    "ExperimentConfig",
    "FrequentistResult",
    "FunnelStageResult",
    "PowerAnalysisResult",
    "PriorSensitivityResult",
    "SRMResult",
    "SegmentResult",
    "UserEventRecord",
    "calculate_sample_size",
    "check_srm",
    "clean_and_validate_data",
    "compute_cohens_h",
    "compute_hdi",
    "fmt_currency",
    "fmt_num",
    "fmt_pct",
    "generate_synthetic_ab_data",
    "load_or_generate_data",
    "newcombe_wilson_diff_ci",
    "run_bayesian_analysis",
    "run_frequentist_analysis",
    "run_power_analysis",
    "t",
    "wilson_ci",
]
