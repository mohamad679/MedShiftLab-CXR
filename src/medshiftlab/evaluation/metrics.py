"""Evaluation metrics for MedShiftLab-CXR.

This module implements label-wise metrics for binary and soft-label
multi-label chest X-ray evaluation. It does not perform model inference,
threshold tuning, calibration fitting, or dataset loading.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score


class BinaryLabelMetrics(BaseModel):
    """Metrics for one project-level label."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    n_available: int = Field(ge=0)
    n_binary: int = Field(ge=0)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    threshold: float = Field(ge=0.0, le=1.0)
    auroc: float | None = None
    auprc: float | None = None
    brier_score: float | None = None
    ece: float | None = None
    f1: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    true_positive: int | None = Field(default=None, ge=0)
    false_positive: int | None = Field(default=None, ge=0)
    true_negative: int | None = Field(default=None, ge=0)
    false_negative: int | None = Field(default=None, ge=0)


class CalibrationBin(BaseModel):
    """One equal-width calibration bin for a label."""

    model_config = ConfigDict(extra="forbid")

    bin_index: int = Field(ge=0)
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    includes_upper_bound: bool
    count: int = Field(ge=0)
    mean_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_target: float | None = Field(default=None, ge=0.0, le=1.0)
    absolute_gap: float | None = Field(default=None, ge=0.0, le=1.0)


class LabelCalibrationSummary(BaseModel):
    """ECE-compatible calibration curve data for one label."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    n_available: int = Field(ge=0)
    n_bins: int = Field(gt=0)
    ece: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bins: list[CalibrationBin]


def evaluate_binary_label(
    label_name: str,
    y_true: Sequence[float | int | None],
    y_score: Sequence[float | int | None],
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> BinaryLabelMetrics:
    """Evaluate one binary/soft-label task.

    Missing values are removed. Soft labels in [0, 1] are used for Brier score
    and ECE. Discrimination and thresholded metrics use only binary targets
    equal to 0 or 1.
    """

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    available_targets: list[float] = []
    available_scores: list[float] = []

    for target, score in zip(y_true, y_score, strict=True):
        if _is_missing(target) or _is_missing(score):
            continue

        parsed_target = _coerce_probability(target, field_name="target")
        parsed_score = _coerce_probability(score, field_name="score")

        available_targets.append(parsed_target)
        available_scores.append(parsed_score)

    n_available = len(available_targets)

    binary_pairs = [
        (target, score)
        for target, score in zip(available_targets, available_scores, strict=True)
        if target in {0.0, 1.0}
    ]

    binary_targets = [target for target, _ in binary_pairs]
    binary_scores = [score for _, score in binary_pairs]

    n_positive = sum(target == 1.0 for target in binary_targets)
    n_negative = sum(target == 0.0 for target in binary_targets)

    auroc = _safe_auroc(binary_targets, binary_scores)
    auprc = _safe_auprc(binary_targets, binary_scores)
    brier_score = _brier_score(available_targets, available_scores)
    ece = _expected_calibration_error(
        available_targets,
        available_scores,
        n_bins=n_bins,
    )

    (
        f1,
        sensitivity,
        specificity,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
    ) = _threshold_metrics(
        binary_targets,
        binary_scores,
        threshold=threshold,
    )

    return BinaryLabelMetrics(
        label_name=label_name,
        n_available=n_available,
        n_binary=len(binary_targets),
        n_positive=n_positive,
        n_negative=n_negative,
        threshold=threshold,
        auroc=auroc,
        auprc=auprc,
        brier_score=brier_score,
        ece=ece,
        f1=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def evaluate_label_metrics(
    y_true_by_label: Mapping[str, Sequence[float | int | None]],
    y_score_by_label: Mapping[str, Sequence[float | int | None]],
    *,
    labels: Iterable[str] | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, BinaryLabelMetrics]:
    """Evaluate multiple labels from target and score mappings."""

    selected_labels = tuple(labels) if labels is not None else tuple(y_true_by_label)

    results: dict[str, BinaryLabelMetrics] = {}

    for label_name in selected_labels:
        if label_name not in y_true_by_label:
            raise ValueError(f"Missing y_true values for label: {label_name}")
        if label_name not in y_score_by_label:
            raise ValueError(f"Missing y_score values for label: {label_name}")

        results[label_name] = evaluate_binary_label(
            label_name,
            y_true_by_label[label_name],
            y_score_by_label[label_name],
            threshold=threshold,
            n_bins=n_bins,
        )

    return results


def summarize_label_calibration(
    label_name: str,
    y_true: Sequence[float | int | None],
    y_score: Sequence[float | int | None],
    *,
    n_bins: int = 10,
) -> LabelCalibrationSummary:
    """Build deterministic equal-width calibration bins for one label.

    Empty bins are retained with null means so exported curve data records the
    complete binning scheme. A score of exactly 1.0 belongs to the final bin.
    """

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    targets: list[float] = []
    scores: list[float] = []
    for target, score in zip(y_true, y_score, strict=True):
        if _is_missing(target) or _is_missing(score):
            continue
        targets.append(_coerce_probability(target, field_name="target"))
        scores.append(_coerce_probability(score, field_name="score"))

    counts = [0] * n_bins
    target_sums = [0.0] * n_bins
    score_sums = [0.0] * n_bins
    for target, score in zip(targets, scores, strict=True):
        bin_index = min(int(score * n_bins), n_bins - 1)
        counts[bin_index] += 1
        target_sums[bin_index] += target
        score_sums[bin_index] += score

    bins: list[CalibrationBin] = []
    for bin_index, count in enumerate(counts):
        mean_target = target_sums[bin_index] / count if count else None
        mean_score = score_sums[bin_index] / count if count else None
        bins.append(
            CalibrationBin(
                bin_index=bin_index,
                lower_bound=bin_index / n_bins,
                upper_bound=(bin_index + 1) / n_bins,
                includes_upper_bound=bin_index == n_bins - 1,
                count=count,
                mean_score=mean_score,
                mean_target=mean_target,
                absolute_gap=(
                    abs(mean_score - mean_target)
                    if mean_score is not None and mean_target is not None
                    else None
                ),
            )
        )

    total = len(targets)
    ece = (
        sum((item.count / total) * (item.absolute_gap or 0.0) for item in bins)
        if total
        else None
    )
    return LabelCalibrationSummary(
        label_name=label_name,
        n_available=total,
        n_bins=n_bins,
        ece=float(ece) if ece is not None else None,
        brier_score=_brier_score(targets, scores),
        bins=bins,
    )


def _safe_auroc(targets: Sequence[float], scores: Sequence[float]) -> float | None:
    if not _has_both_classes(targets):
        return None
    return float(roc_auc_score(targets, scores))


def _safe_auprc(targets: Sequence[float], scores: Sequence[float]) -> float | None:
    if not _has_both_classes(targets):
        return None
    return float(average_precision_score(targets, scores))


def _brier_score(targets: Sequence[float], scores: Sequence[float]) -> float | None:
    if not targets:
        return None

    squared_errors = [
        (score - target) ** 2
        for target, score in zip(targets, scores, strict=True)
    ]
    return float(sum(squared_errors) / len(squared_errors))


def _expected_calibration_error(
    targets: Sequence[float],
    scores: Sequence[float],
    *,
    n_bins: int,
) -> float | None:
    return summarize_label_calibration(
        "label",
        targets,
        scores,
        n_bins=n_bins,
    ).ece


def _threshold_metrics(
    targets: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    int | None,
    int | None,
    int | None,
    int | None,
]:
    if not targets:
        return None, None, None, None, None, None, None

    predictions = [1.0 if score >= threshold else 0.0 for score in scores]

    true_positive = sum(
        target == 1.0 and prediction == 1.0
        for target, prediction in zip(targets, predictions, strict=True)
    )
    false_positive = sum(
        target == 0.0 and prediction == 1.0
        for target, prediction in zip(targets, predictions, strict=True)
    )
    true_negative = sum(
        target == 0.0 and prediction == 0.0
        for target, prediction in zip(targets, predictions, strict=True)
    )
    false_negative = sum(
        target == 1.0 and prediction == 0.0
        for target, prediction in zip(targets, predictions, strict=True)
    )

    f1_denominator = (2 * true_positive) + false_positive + false_negative
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive

    f1 = (2 * true_positive / f1_denominator) if f1_denominator else None
    sensitivity = (
        true_positive / sensitivity_denominator
        if sensitivity_denominator
        else None
    )
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else None
    )

    return (
        float(f1) if f1 is not None else None,
        float(sensitivity) if sensitivity is not None else None,
        float(specificity) if specificity is not None else None,
        int(true_positive),
        int(false_positive),
        int(true_negative),
        int(false_negative),
    )


def _has_both_classes(targets: Sequence[float]) -> bool:
    return any(target == 1.0 for target in targets) and any(
        target == 0.0 for target in targets
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped in {"", "nan", "none", "null", "na"}

    return False


def _coerce_probability(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or missing: {value!r}") from exc

    if math.isnan(parsed):
        raise ValueError(f"{field_name} must not be NaN after missing filtering")

    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0: {value!r}")

    return parsed
