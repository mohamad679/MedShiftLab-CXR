"""Evaluation report schemas for MedShiftLab-CXR.

This module wraps label-wise metrics into a reproducible report object. It does
not run model inference, tune thresholds, fit calibration, or write files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.evaluation.metrics import BinaryLabelMetrics, evaluate_label_metrics


class EvaluationRunMetadata(BaseModel):
    """Metadata describing one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    split: str | None = None
    uncertainty_strategy: str | None = None
    threshold: float = Field(ge=0.0, le=1.0)
    n_bins: int = Field(gt=0)
    notes: str | None = None


class EvaluationAggregateMetrics(BaseModel):
    """Macro-level aggregate metrics across labels."""

    model_config = ConfigDict(extra="forbid")

    n_labels: int = Field(ge=0)
    mean_auroc: float | None = None
    mean_auprc: float | None = None
    mean_brier_score: float | None = None
    mean_ece: float | None = None
    mean_f1: float | None = None
    mean_sensitivity: float | None = None
    mean_specificity: float | None = None


class EvaluationReport(BaseModel):
    """Full evaluation report for one dataset/model/run configuration."""

    model_config = ConfigDict(extra="forbid")

    metadata: EvaluationRunMetadata
    aggregate_metrics: EvaluationAggregateMetrics
    label_metrics: dict[str, BinaryLabelMetrics]

    def to_flat_rows(self) -> list[dict[str, object]]:
        """Return one flat dictionary per label for CSV-like export."""

        rows: list[dict[str, object]] = []

        for label_name, metrics in self.label_metrics.items():
            row = {
                "dataset_name": self.metadata.dataset_name,
                "model_name": self.metadata.model_name,
                "split": self.metadata.split,
                "uncertainty_strategy": self.metadata.uncertainty_strategy,
                "threshold": self.metadata.threshold,
                "n_bins": self.metadata.n_bins,
                "label_name": label_name,
                "n_available": metrics.n_available,
                "n_binary": metrics.n_binary,
                "n_positive": metrics.n_positive,
                "n_negative": metrics.n_negative,
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
                "brier_score": metrics.brier_score,
                "ece": metrics.ece,
                "f1": metrics.f1,
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
            }
            rows.append(row)

        return rows


def create_evaluation_report(
    *,
    dataset_name: str,
    model_name: str,
    y_true_by_label: Mapping[str, Sequence[float | int | None]],
    y_score_by_label: Mapping[str, Sequence[float | int | None]],
    labels: Iterable[str] | None = None,
    split: str | None = None,
    uncertainty_strategy: str | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    notes: str | None = None,
) -> EvaluationReport:
    """Create a complete evaluation report from label-wise targets and scores."""

    label_metrics = evaluate_label_metrics(
        y_true_by_label=y_true_by_label,
        y_score_by_label=y_score_by_label,
        labels=labels,
        threshold=threshold,
        n_bins=n_bins,
    )

    metadata = EvaluationRunMetadata(
        dataset_name=dataset_name,
        model_name=model_name,
        split=split,
        uncertainty_strategy=uncertainty_strategy,
        threshold=threshold,
        n_bins=n_bins,
        notes=notes,
    )

    return EvaluationReport(
        metadata=metadata,
        aggregate_metrics=summarize_evaluation_metrics(label_metrics),
        label_metrics=label_metrics,
    )


def summarize_evaluation_metrics(
    label_metrics: Mapping[str, BinaryLabelMetrics],
) -> EvaluationAggregateMetrics:
    """Compute macro means across label-wise metrics, ignoring unavailable values."""

    return EvaluationAggregateMetrics(
        n_labels=len(label_metrics),
        mean_auroc=_mean_available(metric.auroc for metric in label_metrics.values()),
        mean_auprc=_mean_available(metric.auprc for metric in label_metrics.values()),
        mean_brier_score=_mean_available(
            metric.brier_score for metric in label_metrics.values()
        ),
        mean_ece=_mean_available(metric.ece for metric in label_metrics.values()),
        mean_f1=_mean_available(metric.f1 for metric in label_metrics.values()),
        mean_sensitivity=_mean_available(
            metric.sensitivity for metric in label_metrics.values()
        ),
        mean_specificity=_mean_available(
            metric.specificity for metric in label_metrics.values()
        ),
    )


def _mean_available(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]

    if not available:
        return None

    return float(sum(available) / len(available))
