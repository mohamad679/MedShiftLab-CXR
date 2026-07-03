"""In-memory orchestration of prediction and evaluation contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medshiftlab.evaluation.report import EvaluationReport
from medshiftlab.models.adapter import CXRModelAdapter
from medshiftlab.models.evaluation_bridge import (
    build_evaluation_rows_from_records_and_predictions,
    create_evaluation_report_from_records_and_predictions,
)
from medshiftlab.models.prediction import PredictionBatch


class _AttributeRecord(Protocol):
    image_id: str
    labels: Mapping[str, float | None]


@dataclass(frozen=True)
class _NormalizedRecord:
    image_id: str
    image_path: str | None
    dataset_name: str
    labels: Mapping[str, float | None]


class InMemoryExperimentConfig(BaseModel):
    """Validated configuration for one in-memory evaluation run."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)
    split: str | None = None
    uncertainty_strategy: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    n_bins: int = Field(default=10, gt=0)
    notes: str | None = None

    @field_validator("dataset_name")
    @classmethod
    def _strip_dataset_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset_name must not be empty")
        return value

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if any(not label.strip() for label in labels):
            raise ValueError("labels must contain only non-empty strings")
        return labels


class InMemoryExperimentResult(BaseModel):
    """Predictions, joined rows, and report produced by an in-memory run."""

    model_config = ConfigDict(extra="forbid")

    config: InMemoryExperimentConfig
    model_name: str = Field(min_length=1)
    n_records: int = Field(ge=0)
    predictions: PredictionBatch
    evaluation_rows: list[dict[str, object]]
    report: EvaluationReport

    @field_validator("model_name")
    @classmethod
    def _strip_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_name must not be empty")
        return value


def run_in_memory_evaluation_experiment(
    records: Sequence[_AttributeRecord | Mapping[str, object]],
    adapter: CXRModelAdapter,
    config: InMemoryExperimentConfig,
    *,
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> InMemoryExperimentResult:
    """Run prediction and evaluation entirely in memory."""

    if not records:
        raise ValueError("records must not be empty")
    if not isinstance(adapter, CXRModelAdapter):
        raise TypeError("adapter must implement the CXRModelAdapter protocol")

    normalized_records = [
        _normalize_record(record, config.dataset_name) for record in records
    ]
    image_records: list[dict[str, object]] = [
        {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "dataset_name": record.dataset_name,
        }
        for record in normalized_records
    ]

    predictions = adapter.predict_records(image_records)
    if not isinstance(predictions, PredictionBatch):
        raise TypeError("adapter.predict_records must return a PredictionBatch")

    evaluation_rows = build_evaluation_rows_from_records_and_predictions(
        records=normalized_records,
        predictions=predictions,
        labels=config.labels,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )
    report = create_evaluation_report_from_records_and_predictions(
        records=normalized_records,
        predictions=predictions,
        labels=config.labels,
        dataset_name=config.dataset_name,
        model_name=predictions.model_name,
        split=config.split,
        uncertainty_strategy=config.uncertainty_strategy,
        threshold=config.threshold,
        n_bins=config.n_bins,
        notes=config.notes,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )

    return InMemoryExperimentResult(
        config=config,
        model_name=predictions.model_name,
        n_records=len(normalized_records),
        predictions=predictions,
        evaluation_rows=evaluation_rows,
        report=report,
    )


def _normalize_record(
    record: _AttributeRecord | Mapping[str, object], fallback_dataset_name: str
) -> _NormalizedRecord:
    image_id = _record_value(record, "image_id")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("each record must contain a non-empty string image_id")

    image_path = _record_value(record, "image_path")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError("record image_path must be a string or None")

    dataset_name = _record_value(record, "dataset_name")
    if dataset_name is None:
        dataset_name = fallback_dataset_name
    if not isinstance(dataset_name, str):
        raise ValueError("record dataset_name must be a string or None")

    labels = _record_value(record, "labels")
    if not isinstance(labels, Mapping):
        raise ValueError("each record must contain a labels mapping")

    return _NormalizedRecord(
        image_id=image_id.strip(),
        image_path=image_path,
        dataset_name=dataset_name,
        labels=cast(Mapping[str, float | None], labels),
    )


def _record_value(
    record: _AttributeRecord | Mapping[str, object], field_name: str
) -> object | None:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)
