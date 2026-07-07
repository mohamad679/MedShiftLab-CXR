"""Calibration, bootstrap, subgroup, and failure-case analysis primitives.

The functions in this module operate on already-produced targets and scores.
They do not load images, initialize models, fit calibrators, or make clinical
interpretations.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.evaluation.metrics import (
    BinaryLabelMetrics,
    LabelCalibrationSummary,
    evaluate_binary_label,
    summarize_label_calibration,
)
from medshiftlab.evaluation.report import EvaluationReport


SUPPORTED_BOOTSTRAP_METRICS = frozenset(
    {
        "auroc",
        "auprc",
        "brier_score",
        "ece",
        "f1",
        "sensitivity",
        "specificity",
    }
)
HIGHER_IS_WORSE_METRICS = frozenset({"brier_score", "ece"})
MISSING_GROUP_VALUE = "__missing__"


class BootstrapMetricInterval(BaseModel):
    """Percentile bootstrap interval for one scalar label metric."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence_level: float = Field(gt=0.0, lt=1.0)
    iterations: int = Field(gt=0)
    valid_iterations: int = Field(ge=0)
    seed: int
    resampling_unit: Literal["sample", "patient"]


class SubgroupMetricResult(BaseModel):
    """Metrics for one label within one subgroup value."""

    model_config = ConfigDict(extra="forbid")

    subgroup_column: str = Field(min_length=1)
    subgroup_value: str = Field(min_length=1)
    group_size: int = Field(ge=0)
    metrics: BinaryLabelMetrics


class SkippedSubgroupLabel(BaseModel):
    """A subgroup/label combination omitted from metric computation."""

    model_config = ConfigDict(extra="forbid")

    subgroup_column: str = Field(min_length=1)
    subgroup_value: str = Field(min_length=1)
    label_name: str = Field(min_length=1)
    group_size: int = Field(ge=0)
    n_available: int = Field(ge=0)
    reason: str = Field(min_length=1)


class SubgroupCoverage(BaseModel):
    """Coverage accounting for one requested subgroup column."""

    model_config = ConfigDict(extra="forbid")

    subgroup_column: str = Field(min_length=1)
    total_rows: int = Field(ge=0)
    missing_rows: int = Field(ge=0)
    distinct_values: int = Field(ge=0)


class SubgroupAnalysisReport(BaseModel):
    """Typed subgroup metrics, skip reasons, and metadata coverage."""

    model_config = ConfigDict(extra="forbid")

    subgroup_columns: tuple[str, ...]
    minimum_subgroup_size: int = Field(gt=0)
    results: list[SubgroupMetricResult]
    skipped: list[SkippedSubgroupLabel]
    coverage: list[SubgroupCoverage]


class MetricDegradation(BaseModel):
    """A label metric degraded relative to a supplied baseline report."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    current_value: float
    baseline_value: float
    degradation: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)


class PoorCalibrationFlag(BaseModel):
    """A calibration metric exceeding an explicit threshold."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    metric_name: Literal["ece", "brier_score"]
    value: float
    threshold: float = Field(ge=0.0)


class CoverageIssue(BaseModel):
    """A non-clinical coverage or analyzability issue."""

    model_config = ConfigDict(extra="forbid")

    issue_type: str = Field(min_length=1)
    subgroup_column: str | None = None
    subgroup_value: str | None = None
    label_name: str | None = None
    detail: str = Field(min_length=1)


class FailureCaseSummary(BaseModel):
    """Machine-readable flags; these are not clinical failure conclusions."""

    model_config = ConfigDict(extra="forbid")

    metric_degradations: list[MetricDegradation]
    poor_calibration: list[PoorCalibrationFlag]
    coverage_issues: list[CoverageIssue]
    clinical_interpretation_performed: bool = False


class RobustnessAnalysisReport(BaseModel):
    """Complete Phase 10 analysis artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["medshiftlab.robustness.v1"] = "medshiftlab.robustness.v1"
    dataset_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    evaluated_records: int = Field(ge=0)
    calibration: dict[str, LabelCalibrationSummary]
    bootstrap_intervals: list[BootstrapMetricInterval]
    subgroup_analysis: SubgroupAnalysisReport
    failure_cases: FailureCaseSummary
    support_status: dict[str, str]
    manual_local_only: bool = True
    real_inference_performed: bool = False


def bootstrap_label_metric_intervals(
    label_name: str,
    y_true: Sequence[float | int | None],
    y_score: Sequence[float | int | None],
    *,
    metrics: Iterable[str],
    iterations: int,
    seed: int,
    threshold: float = 0.5,
    n_bins: int = 10,
    confidence_level: float = 0.95,
    patient_ids: Sequence[str | None] | None = None,
) -> list[BootstrapMetricInterval]:
    """Compute deterministic percentile intervals by sample or patient cluster."""

    metric_names = tuple(metrics)
    unsupported = sorted(set(metric_names) - SUPPORTED_BOOTSTRAP_METRICS)
    if unsupported:
        raise ValueError(
            "Unsupported bootstrap metric(s): "
            + ", ".join(unsupported)
            + ". Supported metrics: "
            + ", ".join(sorted(SUPPORTED_BOOTSTRAP_METRICS))
        )
    if not metric_names:
        raise ValueError("At least one bootstrap metric is required")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if patient_ids is not None and len(patient_ids) != len(y_true):
        raise ValueError("patient_ids must align with y_true and y_score")
    if not y_true:
        raise ValueError("bootstrap inputs must not be empty")

    use_patient_clusters = patient_ids is not None and all(patient_ids)
    if use_patient_clusters:
        indexes_by_patient: dict[str, list[int]] = defaultdict(list)
        for index, patient_id in enumerate(patient_ids or ()):
            indexes_by_patient[str(patient_id)].append(index)
        units: list[list[int]] = list(indexes_by_patient.values())
        resampling_unit: Literal["sample", "patient"] = "patient"
    else:
        units = [[index] for index in range(len(y_true))]
        resampling_unit = "sample"

    rng = random.Random(seed)
    values_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sampled_indexes: list[int] = []
        for _unit_index in range(len(units)):
            sampled_indexes.extend(units[rng.randrange(len(units))])
        sampled_metrics = evaluate_binary_label(
            label_name,
            [y_true[index] for index in sampled_indexes],
            [y_score[index] for index in sampled_indexes],
            threshold=threshold,
            n_bins=n_bins,
        )
        for metric_name in metric_names:
            value = getattr(sampled_metrics, metric_name)
            if value is not None:
                values_by_metric[metric_name].append(float(value))

    point_metrics = evaluate_binary_label(
        label_name,
        y_true,
        y_score,
        threshold=threshold,
        n_bins=n_bins,
    )
    alpha = (1.0 - confidence_level) / 2.0
    intervals: list[BootstrapMetricInterval] = []
    for metric_name in metric_names:
        values = values_by_metric[metric_name]
        intervals.append(
            BootstrapMetricInterval(
                label_name=label_name,
                metric_name=metric_name,
                point_estimate=getattr(point_metrics, metric_name),
                lower=_percentile(values, alpha) if values else None,
                upper=_percentile(values, 1.0 - alpha) if values else None,
                confidence_level=confidence_level,
                iterations=iterations,
                valid_iterations=len(values),
                seed=seed,
                resampling_unit=resampling_unit,
            )
        )
    return intervals


def analyze_subgroups(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Iterable[str],
    subgroup_columns: Iterable[str],
    minimum_subgroup_size: int,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> SubgroupAnalysisReport:
    """Compute per-label metrics for requested metadata subgroups."""

    if not rows:
        raise ValueError("subgroup rows must not be empty")
    if minimum_subgroup_size <= 0:
        raise ValueError("minimum_subgroup_size must be positive")
    label_names = tuple(labels)
    columns = tuple(subgroup_columns)
    if not label_names:
        raise ValueError("labels must not be empty")
    if not columns:
        raise ValueError("subgroup_columns must not be empty")

    missing_columns = [
        column for column in columns if not any(column in row for row in rows)
    ]
    if missing_columns:
        raise ValueError(
            "Missing requested subgroup column(s): " + ", ".join(missing_columns)
        )

    results: list[SubgroupMetricResult] = []
    skipped: list[SkippedSubgroupLabel] = []
    coverage: list[SubgroupCoverage] = []
    for column in columns:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        missing_rows = 0
        for row in rows:
            value = _normalize_group_value(row.get(column))
            if value == MISSING_GROUP_VALUE:
                missing_rows += 1
            groups[value].append(row)
        coverage.append(
            SubgroupCoverage(
                subgroup_column=column,
                total_rows=len(rows),
                missing_rows=missing_rows,
                distinct_values=len(groups),
            )
        )

        for value in sorted(groups):
            group_rows = groups[value]
            for label_name in label_names:
                target_column = f"true_{label_name}"
                score_column = f"score_{label_name}"
                if any(
                    target_column not in row or score_column not in row
                    for row in group_rows
                ):
                    raise ValueError(
                        f"Subgroup rows are missing {target_column!r} or "
                        f"{score_column!r}"
                    )
                targets = [row[target_column] for row in group_rows]
                scores = [row[score_column] for row in group_rows]
                n_available = sum(
                    not _is_missing(target) and not _is_missing(score)
                    for target, score in zip(targets, scores, strict=True)
                )
                if n_available < minimum_subgroup_size:
                    skipped.append(
                        SkippedSubgroupLabel(
                            subgroup_column=column,
                            subgroup_value=value,
                            label_name=label_name,
                            group_size=len(group_rows),
                            n_available=n_available,
                            reason="insufficient_available_samples",
                        )
                    )
                    continue
                results.append(
                    SubgroupMetricResult(
                        subgroup_column=column,
                        subgroup_value=value,
                        group_size=len(group_rows),
                        metrics=evaluate_binary_label(
                            label_name,
                            targets,
                            scores,
                            threshold=threshold,
                            n_bins=n_bins,
                        ),
                    )
                )

    return SubgroupAnalysisReport(
        subgroup_columns=columns,
        minimum_subgroup_size=minimum_subgroup_size,
        results=results,
        skipped=skipped,
        coverage=coverage,
    )


def summarize_failure_cases(
    current_report: EvaluationReport,
    *,
    subgroup_report: SubgroupAnalysisReport,
    baseline_report: EvaluationReport | None = None,
    degradation_metric: str = "auroc",
    degradation_threshold: float | None = None,
    maximum_ece: float | None = None,
    maximum_brier_score: float | None = None,
) -> FailureCaseSummary:
    """Flag explicit metric, calibration, and coverage conditions."""

    if degradation_metric not in SUPPORTED_BOOTSTRAP_METRICS:
        raise ValueError(f"Unsupported degradation metric: {degradation_metric}")
    for name, value in (
        ("degradation_threshold", degradation_threshold),
        ("maximum_ece", maximum_ece),
        ("maximum_brier_score", maximum_brier_score),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")

    degradations: list[MetricDegradation] = []
    coverage_issues: list[CoverageIssue] = []
    if baseline_report is not None and degradation_threshold is not None:
        for label_name, current in current_report.label_metrics.items():
            baseline = baseline_report.label_metrics.get(label_name)
            if baseline is None:
                coverage_issues.append(
                    CoverageIssue(
                        issue_type="baseline_label_missing",
                        label_name=label_name,
                        detail="Comparison baseline does not contain this label.",
                    )
                )
                continue
            current_value = getattr(current, degradation_metric)
            baseline_value = getattr(baseline, degradation_metric)
            if current_value is None or baseline_value is None:
                coverage_issues.append(
                    CoverageIssue(
                        issue_type="comparison_metric_unavailable",
                        label_name=label_name,
                        detail=f"{degradation_metric} is unavailable in current or baseline report.",
                    )
                )
                continue
            degradation = (
                current_value - baseline_value
                if degradation_metric in HIGHER_IS_WORSE_METRICS
                else baseline_value - current_value
            )
            if degradation > 0.0 and degradation >= degradation_threshold:
                degradations.append(
                    MetricDegradation(
                        label_name=label_name,
                        metric_name=degradation_metric,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        degradation=degradation,
                        threshold=degradation_threshold,
                    )
                )

    poor_calibration: list[PoorCalibrationFlag] = []
    for label_name, metrics in current_report.label_metrics.items():
        if maximum_ece is not None and metrics.ece is not None and metrics.ece > maximum_ece:
            poor_calibration.append(
                PoorCalibrationFlag(
                    label_name=label_name,
                    metric_name="ece",
                    value=metrics.ece,
                    threshold=maximum_ece,
                )
            )
        if (
            maximum_brier_score is not None
            and metrics.brier_score is not None
            and metrics.brier_score > maximum_brier_score
        ):
            poor_calibration.append(
                PoorCalibrationFlag(
                    label_name=label_name,
                    metric_name="brier_score",
                    value=metrics.brier_score,
                    threshold=maximum_brier_score,
                )
            )

    for item in subgroup_report.skipped:
        coverage_issues.append(
            CoverageIssue(
                issue_type="subgroup_label_skipped",
                subgroup_column=item.subgroup_column,
                subgroup_value=item.subgroup_value,
                label_name=item.label_name,
                detail=item.reason,
            )
        )
    for item in subgroup_report.coverage:
        if item.missing_rows:
            coverage_issues.append(
                CoverageIssue(
                    issue_type="missing_subgroup_metadata",
                    subgroup_column=item.subgroup_column,
                    detail=f"{item.missing_rows} of {item.total_rows} rows are missing metadata.",
                )
            )

    return FailureCaseSummary(
        metric_degradations=degradations,
        poor_calibration=poor_calibration,
        coverage_issues=coverage_issues,
    )


def build_calibration_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Iterable[str],
    n_bins: int,
) -> dict[str, LabelCalibrationSummary]:
    """Build per-label calibration summaries from evaluation-compatible rows."""

    return {
        label: summarize_label_calibration(
            label,
            [row[f"true_{label}"] for row in rows],
            [row[f"score_{label}"] for row in rows],
            n_bins=n_bins,
        )
        for label in labels
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(
        ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
    )


def _normalize_group_value(value: Any) -> str:
    if _is_missing(value):
        return MISSING_GROUP_VALUE
    normalized = str(value).strip()
    if normalized.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", normalized):
        raise ValueError("Absolute paths are not allowed in subgroup metadata values")
    return normalized


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "na"}
    return False
