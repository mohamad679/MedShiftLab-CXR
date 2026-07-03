"""Evaluation utilities for MedShiftLab-CXR."""

from medshiftlab.evaluation.metrics import (
    BinaryLabelMetrics,
    evaluate_binary_label,
    evaluate_label_metrics,
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

__all__ = [
    "BinaryLabelMetrics",
    "EvaluationAggregateMetrics",
    "EvaluationReport",
    "EvaluationRunMetadata",
    "build_label_matrices_from_rows",
    "create_evaluation_report",
    "create_evaluation_report_from_rows",
    "evaluate_binary_label",
    "evaluate_label_metrics",
    "summarize_evaluation_metrics",
]
