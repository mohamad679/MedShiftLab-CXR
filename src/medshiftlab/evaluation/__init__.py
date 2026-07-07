"""Evaluation utilities for MedShiftLab-CXR."""

from medshiftlab.evaluation.metrics import (
    BinaryLabelMetrics,
    CalibrationBin,
    LabelCalibrationSummary,
    evaluate_binary_label,
    evaluate_label_metrics,
    summarize_label_calibration,
)
from medshiftlab.evaluation.report import (
    EvaluationAggregateMetrics,
    EvaluationReport,
    EvaluationRunMetadata,
    create_evaluation_report,
    summarize_evaluation_metrics,
)
from medshiftlab.evaluation.table import (
    build_label_matrices_from_rows,
    create_evaluation_report_from_rows,
)
from medshiftlab.evaluation.robustness import (
    BootstrapMetricInterval,
    FailureCaseSummary,
    RobustnessAnalysisReport,
    SubgroupAnalysisReport,
    analyze_subgroups,
    bootstrap_label_metric_intervals,
    summarize_failure_cases,
)

__all__ = [
    "BinaryLabelMetrics",
    "BootstrapMetricInterval",
    "CalibrationBin",
    "EvaluationAggregateMetrics",
    "EvaluationReport",
    "EvaluationRunMetadata",
    "FailureCaseSummary",
    "LabelCalibrationSummary",
    "RobustnessAnalysisReport",
    "SubgroupAnalysisReport",
    "analyze_subgroups",
    "bootstrap_label_metric_intervals",
    "build_label_matrices_from_rows",
    "create_evaluation_report",
    "create_evaluation_report_from_rows",
    "evaluate_binary_label",
    "evaluate_label_metrics",
    "summarize_failure_cases",
    "summarize_label_calibration",
    "summarize_evaluation_metrics",
]
