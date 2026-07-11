"""Cross-dataset bootstrap confidence intervals for existing predictions.

This module operates only on standardized prediction and label artifacts. It
does not load images, initialize models, run inference, train, calibrate, tune
thresholds, or alter metric formulas.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.evaluation.metrics import evaluate_binary_label
from medshiftlab.experiments.prediction_evaluation import (
    GroundTruthLabelRow,
    load_ground_truth_label_rows,
    load_prediction_batch,
)
from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


CROSS_DATASET_BOOTSTRAP_SCHEMA_VERSION = "medshiftlab.cross_dataset_bootstrap.v1"
DELTA_DEFINITION = "external_minus_reference"
SUPPORTED_CROSS_DATASET_BOOTSTRAP_METRICS = frozenset(
    {"auroc", "auprc", "brier_score", "ece"}
)


class DatasetBootstrapSummary(BaseModel):
    """Bootstrap summary for one dataset, label, and metric."""

    model_config = ConfigDict(extra="forbid")

    point_estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    valid_iterations: int = Field(ge=0)


class CrossDatasetBootstrapMetricResult(BaseModel):
    """Aggregate cross-dataset bootstrap result for one label metric."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    metric_name: Literal["auroc", "auprc", "brier_score", "ece"]
    metric_direction: Literal["higher_is_better", "lower_is_better"]
    reference_point_estimate: float | None = None
    reference_ci_lower: float | None = None
    reference_ci_upper: float | None = None
    reference_valid_iterations: int = Field(ge=0)
    external_point_estimate: float | None = None
    external_ci_lower: float | None = None
    external_ci_upper: float | None = None
    external_valid_iterations: int = Field(ge=0)
    delta_point_estimate: float | None = None
    delta_ci_lower: float | None = None
    delta_ci_upper: float | None = None
    delta_valid_iterations: int = Field(ge=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    iterations: int = Field(gt=0)


class CrossDatasetBootstrapReport(BaseModel):
    """Aggregate cross-dataset bootstrap report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "medshiftlab.cross_dataset_bootstrap.v1"
    ] = CROSS_DATASET_BOOTSTRAP_SCHEMA_VERSION
    reference_dataset_name: str = Field(min_length=1)
    external_dataset_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    shared_labels: tuple[str, ...] = Field(min_length=1)
    requested_metrics: tuple[Literal["auroc", "auprc", "brier_score", "ece"], ...] = (
        Field(min_length=1)
    )
    iterations: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    seed: int
    n_bins: int = Field(gt=0)
    reference_resampling_unit: Literal["sample", "patient"]
    external_resampling_unit: Literal["sample", "patient"]
    reference_record_count: int = Field(ge=0)
    external_record_count: int = Field(ge=0)
    results: list[CrossDatasetBootstrapMetricResult]
    manual_local_only: bool = True
    real_inference_performed: bool = False
    clinical_validation_completed: bool = False
    prospective_validation_completed: bool = False
    independent_dataset_bootstrap: bool = True
    delta_definition: Literal["external_minus_reference"] = DELTA_DEFINITION


class _BootstrapDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch: PredictionBatch
    records: list[PredictionRecord]
    label_rows_by_id: dict[str, GroundTruthLabelRow]
    dataset_name: str
    resampling_unit: Literal["sample", "patient"]
    units: list[list[int]]


def run_cross_dataset_bootstrap_from_files(
    *,
    reference_predictions_path: str | Path,
    reference_labels_csv_path: str | Path,
    external_predictions_path: str | Path,
    external_labels_csv_path: str | Path,
    iterations: int = 1000,
    seed: int = 2026,
    confidence_level: float = 0.95,
    n_bins: int = 10,
    metrics: Iterable[str] = ("auroc", "auprc", "brier_score", "ece"),
) -> CrossDatasetBootstrapReport:
    """Run independent cross-dataset bootstrap analysis from existing artifacts."""

    metric_names = _validate_inputs(
        iterations=iterations,
        confidence_level=confidence_level,
        n_bins=n_bins,
        metrics=metrics,
    )

    reference_batch, _ = load_prediction_batch(reference_predictions_path)
    external_batch, _ = load_prediction_batch(external_predictions_path)
    if reference_batch.model_name != external_batch.model_name:
        raise ValueError(
            "Reference and external prediction batches must have the same model_name"
        )

    shared_labels = tuple(
        label_name
        for label_name in reference_batch.label_names
        if label_name in set(external_batch.label_names)
    )
    if not shared_labels:
        raise ValueError("Reference and external prediction batches have no shared labels")

    reference_label_rows = load_ground_truth_label_rows(
        reference_labels_csv_path,
        label_names=shared_labels,
    )
    external_label_rows = load_ground_truth_label_rows(
        external_labels_csv_path,
        label_names=shared_labels,
    )

    reference_dataset = _prepare_bootstrap_dataset(
        reference_batch,
        reference_label_rows,
    )
    external_dataset = _prepare_bootstrap_dataset(
        external_batch,
        external_label_rows,
    )

    reference_iteration_indexes = _sample_iteration_indexes(
        reference_dataset.units,
        iterations=iterations,
        seed=_derive_seed(seed, "reference"),
    )
    external_iteration_indexes = _sample_iteration_indexes(
        external_dataset.units,
        iterations=iterations,
        seed=_derive_seed(seed, "external"),
    )

    results: list[CrossDatasetBootstrapMetricResult] = []
    for label_name in shared_labels:
        reference_targets, reference_scores = _label_vectors(
            reference_dataset,
            label_name,
        )
        external_targets, external_scores = _label_vectors(
            external_dataset,
            label_name,
        )
        reference_point_metrics = evaluate_binary_label(
            label_name,
            reference_targets,
            reference_scores,
            n_bins=n_bins,
        )
        external_point_metrics = evaluate_binary_label(
            label_name,
            external_targets,
            external_scores,
            n_bins=n_bins,
        )

        bootstrap_values = _bootstrap_metric_values(
            label_name=label_name,
            metrics=metric_names,
            reference_targets=reference_targets,
            reference_scores=reference_scores,
            external_targets=external_targets,
            external_scores=external_scores,
            reference_iteration_indexes=reference_iteration_indexes,
            external_iteration_indexes=external_iteration_indexes,
            n_bins=n_bins,
        )

        for metric_name in metric_names:
            reference_values = bootstrap_values["reference"][metric_name]
            external_values = bootstrap_values["external"][metric_name]
            delta_values = bootstrap_values["delta"][metric_name]
            reference_summary = _summarize_dataset_bootstrap(
                getattr(reference_point_metrics, metric_name),
                reference_values,
                confidence_level=confidence_level,
            )
            external_summary = _summarize_dataset_bootstrap(
                getattr(external_point_metrics, metric_name),
                external_values,
                confidence_level=confidence_level,
            )
            delta_point_estimate = _subtract_if_available(
                external_summary.point_estimate,
                reference_summary.point_estimate,
            )
            alpha = (1.0 - confidence_level) / 2.0
            results.append(
                CrossDatasetBootstrapMetricResult(
                    label_name=label_name,
                    metric_name=metric_name,  # type: ignore[arg-type]
                    metric_direction=_metric_direction(metric_name),
                    reference_point_estimate=reference_summary.point_estimate,
                    reference_ci_lower=reference_summary.ci_lower,
                    reference_ci_upper=reference_summary.ci_upper,
                    reference_valid_iterations=reference_summary.valid_iterations,
                    external_point_estimate=external_summary.point_estimate,
                    external_ci_lower=external_summary.ci_lower,
                    external_ci_upper=external_summary.ci_upper,
                    external_valid_iterations=external_summary.valid_iterations,
                    delta_point_estimate=delta_point_estimate,
                    delta_ci_lower=(
                        _percentile(delta_values, alpha) if delta_values else None
                    ),
                    delta_ci_upper=(
                        _percentile(delta_values, 1.0 - alpha)
                        if delta_values
                        else None
                    ),
                    delta_valid_iterations=len(delta_values),
                    confidence_level=confidence_level,
                    iterations=iterations,
                )
            )

    return CrossDatasetBootstrapReport(
        reference_dataset_name=reference_dataset.dataset_name,
        external_dataset_name=external_dataset.dataset_name,
        model_name=reference_batch.model_name,
        shared_labels=shared_labels,
        requested_metrics=metric_names,  # type: ignore[arg-type]
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
        n_bins=n_bins,
        reference_resampling_unit=reference_dataset.resampling_unit,
        external_resampling_unit=external_dataset.resampling_unit,
        reference_record_count=len(reference_dataset.records),
        external_record_count=len(external_dataset.records),
        results=results,
    )


def _validate_inputs(
    *,
    iterations: int,
    confidence_level: float,
    n_bins: int,
    metrics: Iterable[str],
) -> tuple[str, ...]:
    metric_names = tuple(dict.fromkeys(item.strip() for item in metrics if item.strip()))
    unsupported = sorted(set(metric_names) - SUPPORTED_CROSS_DATASET_BOOTSTRAP_METRICS)
    if unsupported:
        raise ValueError(
            "Unsupported cross-dataset bootstrap metric(s): "
            + ", ".join(unsupported)
            + ". Supported metrics: "
            + ", ".join(sorted(SUPPORTED_CROSS_DATASET_BOOTSTRAP_METRICS))
        )
    if not metric_names:
        raise ValueError("At least one supported cross-dataset bootstrap metric is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return metric_names


def _prepare_bootstrap_dataset(
    batch: PredictionBatch,
    label_rows: list[GroundTruthLabelRow],
) -> _BootstrapDataset:
    records = batch.records
    dataset_names = {record.dataset_name for record in records}
    if len(dataset_names) != 1:
        raise ValueError(
            "Prediction batch must contain exactly one dataset_name for bootstrap"
        )

    label_rows_by_id: dict[str, GroundTruthLabelRow] = {}
    duplicate_label_ids = 0
    for row in label_rows:
        if row.sample_id in label_rows_by_id:
            duplicate_label_ids += 1
            continue
        label_rows_by_id[row.sample_id] = row
    if duplicate_label_ids:
        raise ValueError(
            "Ground-truth labels CSV contains duplicate sample_id values: "
            f"{duplicate_label_ids}"
        )

    prediction_ids = {record.sample_id for record in records}
    label_ids = set(label_rows_by_id)
    missing_labels = len(prediction_ids - label_ids)
    missing_predictions = len(label_ids - prediction_ids)
    if missing_labels or missing_predictions:
        raise ValueError(
            "Prediction/label sample_id mismatch: "
            f"missing_labels={missing_labels}, missing_predictions={missing_predictions}"
        )

    if all(record.patient_id for record in records):
        indexes_by_patient: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            indexes_by_patient[str(record.patient_id)].append(index)
        units = list(indexes_by_patient.values())
        resampling_unit: Literal["sample", "patient"] = "patient"
    else:
        units = [[index] for index in range(len(records))]
        resampling_unit = "sample"

    return _BootstrapDataset(
        batch=batch,
        records=records,
        label_rows_by_id=label_rows_by_id,
        dataset_name=next(iter(dataset_names)),
        resampling_unit=resampling_unit,
        units=units,
    )


def _sample_iteration_indexes(
    units: Sequence[Sequence[int]],
    *,
    iterations: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    sampled: list[list[int]] = []
    for _ in range(iterations):
        iteration_indexes: list[int] = []
        for _unit_index in range(len(units)):
            iteration_indexes.extend(units[rng.randrange(len(units))])
        sampled.append(iteration_indexes)
    return sampled


def _label_vectors(
    dataset: _BootstrapDataset,
    label_name: str,
) -> tuple[list[float | int | None], list[float | int | None]]:
    targets: list[float | int | None] = []
    scores: list[float | int | None] = []
    for record in dataset.records:
        label_row = dataset.label_rows_by_id[record.sample_id]
        if label_name not in label_row.model_extra:
            raise ValueError(
                f"Ground-truth label row for sample_id {record.sample_id!r} "
                f"is missing label {label_name!r}"
            )
        targets.append(label_row.model_extra[label_name])
        scores.append(record.scores[label_name])
    return targets, scores


def _bootstrap_metric_values(
    *,
    label_name: str,
    metrics: tuple[str, ...],
    reference_targets: Sequence[float | int | None],
    reference_scores: Sequence[float | int | None],
    external_targets: Sequence[float | int | None],
    external_scores: Sequence[float | int | None],
    reference_iteration_indexes: Sequence[Sequence[int]],
    external_iteration_indexes: Sequence[Sequence[int]],
    n_bins: int,
) -> dict[str, dict[str, list[float]]]:
    values = {
        "reference": {metric_name: [] for metric_name in metrics},
        "external": {metric_name: [] for metric_name in metrics},
        "delta": {metric_name: [] for metric_name in metrics},
    }

    for reference_indexes, external_indexes in zip(
        reference_iteration_indexes,
        external_iteration_indexes,
        strict=True,
    ):
        reference_metrics = evaluate_binary_label(
            label_name,
            [reference_targets[index] for index in reference_indexes],
            [reference_scores[index] for index in reference_indexes],
            n_bins=n_bins,
        )
        external_metrics = evaluate_binary_label(
            label_name,
            [external_targets[index] for index in external_indexes],
            [external_scores[index] for index in external_indexes],
            n_bins=n_bins,
        )
        for metric_name in metrics:
            reference_value = getattr(reference_metrics, metric_name)
            external_value = getattr(external_metrics, metric_name)
            if reference_value is not None:
                values["reference"][metric_name].append(float(reference_value))
            if external_value is not None:
                values["external"][metric_name].append(float(external_value))
            delta_value = _subtract_if_available(external_value, reference_value)
            if delta_value is not None:
                values["delta"][metric_name].append(delta_value)
    return values


def _summarize_dataset_bootstrap(
    point_estimate: float | int | None,
    values: Sequence[float],
    *,
    confidence_level: float,
) -> DatasetBootstrapSummary:
    alpha = (1.0 - confidence_level) / 2.0
    return DatasetBootstrapSummary(
        point_estimate=float(point_estimate) if point_estimate is not None else None,
        ci_lower=_percentile(values, alpha) if values else None,
        ci_upper=_percentile(values, 1.0 - alpha) if values else None,
        valid_iterations=len(values),
    )


def _subtract_if_available(
    external_value: float | int | None,
    reference_value: float | int | None,
) -> float | None:
    if external_value is None or reference_value is None:
        return None
    return float(external_value) - float(reference_value)


def _metric_direction(metric_name: str) -> Literal["higher_is_better", "lower_is_better"]:
    if metric_name in {"auroc", "auprc"}:
        return "higher_is_better"
    return "lower_is_better"


def _derive_seed(seed: int, stream_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


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
