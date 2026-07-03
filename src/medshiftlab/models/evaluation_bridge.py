"""Bridge validated data records and prediction batches to evaluation rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from medshiftlab.evaluation.report import EvaluationReport
from medshiftlab.evaluation.table import create_evaluation_report_from_rows
from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


class _EvaluationRecord(Protocol):
    image_id: str
    labels: Mapping[str, float | None]


def build_evaluation_rows_from_records_and_predictions(
    records: Sequence[_EvaluationRecord],
    predictions: PredictionBatch,
    labels: Sequence[str],
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> list[dict[str, object]]:
    """Join data records and predictions by image ID in data-record order."""

    label_list = tuple(labels)
    if not label_list:
        raise ValueError("labels must not be empty")

    record_ids = _unique_record_ids(records)
    predictions_by_id = _predictions_by_image_id(predictions.records)
    prediction_ids = set(predictions_by_id)

    missing_predictions = record_ids - prediction_ids
    if missing_predictions:
        raise ValueError(
            "missing predictions for image_ids: "
            + ", ".join(sorted(missing_predictions))
        )

    unknown_predictions = prediction_ids - record_ids
    if unknown_predictions:
        raise ValueError(
            "predictions contain unknown image_ids: "
            + ", ".join(sorted(unknown_predictions))
        )

    rows: list[dict[str, object]] = []
    for record in records:
        prediction = predictions_by_id[record.image_id]
        missing_targets = [label for label in label_list if label not in record.labels]
        if missing_targets:
            raise ValueError(
                f"data record {record.image_id!r} is missing labels: "
                + ", ".join(missing_targets)
            )
        missing_scores = [label for label in label_list if label not in prediction.scores]
        if missing_scores:
            raise ValueError(
                f"prediction record {prediction.image_id!r} is missing labels: "
                + ", ".join(missing_scores)
            )

        row: dict[str, object] = {
            "image_id": record.image_id,
            "image_path": getattr(record, "image_path", None),
            "dataset_name": getattr(record, "dataset_name", None),
            "model_name": predictions.model_name,
        }
        for label in label_list:
            row[f"{target_prefix}{label}"] = record.labels[label]
            row[f"{score_prefix}{label}"] = prediction.scores[label]
        rows.append(row)

    return rows


def create_evaluation_report_from_records_and_predictions(
    records: Sequence[_EvaluationRecord],
    predictions: PredictionBatch,
    labels: Sequence[str],
    dataset_name: str,
    model_name: str | None = None,
    split: str | None = None,
    uncertainty_strategy: str | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    notes: str | None = None,
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> EvaluationReport:
    """Create an evaluation report from matched data records and predictions."""

    rows = build_evaluation_rows_from_records_and_predictions(
        records=records,
        predictions=predictions,
        labels=labels,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )
    return create_evaluation_report_from_rows(
        rows=rows,
        labels=labels,
        dataset_name=dataset_name,
        model_name=predictions.model_name if model_name is None else model_name,
        split=split,
        uncertainty_strategy=uncertainty_strategy,
        threshold=threshold,
        n_bins=n_bins,
        notes=notes,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )


def _unique_record_ids(records: Sequence[_EvaluationRecord]) -> set[str]:
    image_ids: set[str] = set()
    for record in records:
        if record.image_id in image_ids:
            raise ValueError(f"duplicate data record image_id: {record.image_id!r}")
        image_ids.add(record.image_id)
    return image_ids


def _predictions_by_image_id(
    predictions: Sequence[PredictionRecord],
) -> dict[str, PredictionRecord]:
    by_image_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.image_id in by_image_id:
            raise ValueError(
                f"duplicate prediction image_id: {prediction.image_id!r}"
            )
        by_image_id[prediction.image_id] = prediction
    return by_image_id
