"""Tests for joining MedShiftLab-CXR data records with predictions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from medshiftlab.data import CheXpertRecord
from medshiftlab.evaluation import EvaluationReport
from medshiftlab.models import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBatch,
    PredictionRecord,
    build_evaluation_rows_from_records_and_predictions,
    create_evaluation_report_from_records_and_predictions,
)


LABELS = ("Atelectasis", "Cardiomegaly")


def _record(
    image_id: str,
    atelectasis: float | None = 0.0,
    cardiomegaly: float | None = 1.0,
) -> CheXpertRecord:
    return CheXpertRecord(
        image_id=image_id,
        image_path=f"images/{image_id}.png",
        labels={
            "Atelectasis": atelectasis,
            "Cardiomegaly": cardiomegaly,
        },
        raw_labels={},
    )


def _prediction(
    image_id: str,
    atelectasis: float = 0.2,
    cardiomegaly: float = 0.8,
    *,
    scores: dict[str, float | None] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        image_id=image_id,
        image_path=f"images/{image_id}.png",
        dataset_name="CheXpert",
        model_name="mock-cxr",
        scores=scores
        or {
            "Atelectasis": atelectasis,
            "Cardiomegaly": cardiomegaly,
        },
    )


def _batch(
    predictions: list[PredictionRecord], labels: tuple[str, ...] = LABELS
) -> PredictionBatch:
    return PredictionBatch(
        model_name="mock-cxr",
        model_version="mock-cxr:v1",
        adapter_name="mock-adapter",
        preprocessing_version="preprocess-v1",
        preprocessing_config={"output_mode": "grayscale"},
        records=predictions,
        label_names=labels,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        schema_version=PREDICTION_SCHEMA_VERSION,
    )


def test_build_evaluation_rows_creates_target_and_score_columns() -> None:
    rows = build_evaluation_rows_from_records_and_predictions(
        records=[_record("img001")],
        predictions=_batch([_prediction("img001")]),
        labels=LABELS,
    )

    assert rows == [
        {
            "image_id": "img001",
            "image_path": "images/img001.png",
            "dataset_name": "CheXpert",
            "model_name": "mock-cxr",
            "true_Atelectasis": 0.0,
            "score_Atelectasis": 0.2,
            "true_Cardiomegaly": 1.0,
            "score_Cardiomegaly": 0.8,
        }
    ]


def test_build_evaluation_rows_follows_record_order() -> None:
    rows = build_evaluation_rows_from_records_and_predictions(
        records=[_record("img002"), _record("img001")],
        predictions=_batch([_prediction("img001"), _prediction("img002")]),
        labels=LABELS,
    )

    assert [row["image_id"] for row in rows] == ["img002", "img001"]


def test_create_report_from_records_and_predictions() -> None:
    records = [
        _record("img001", 0.0, 1.0),
        _record("img002", 1.0, 0.0),
        _record("img003", 0.0, 1.0),
        _record("img004", 1.0, 0.0),
    ]
    predictions = _batch(
        [
            _prediction("img003", 0.2, 0.8),
            _prediction("img001", 0.1, 0.9),
            _prediction("img004", 0.8, 0.2),
            _prediction("img002", 0.9, 0.1),
        ]
    )

    report = create_evaluation_report_from_records_and_predictions(
        records=records,
        predictions=predictions,
        labels=LABELS,
        dataset_name="CheXpert",
        split="validation",
        uncertainty_strategy="U-ignore",
        n_bins=2,
        notes="bridge fixture",
    )

    assert isinstance(report, EvaluationReport)
    assert report.metadata.dataset_name == "CheXpert"
    assert report.metadata.model_name == "mock-cxr"
    assert report.metadata.split == "validation"
    assert report.metadata.uncertainty_strategy == "U-ignore"
    assert report.metadata.notes == "bridge fixture"
    assert report.label_metrics["Atelectasis"].auroc == 1.0
    assert report.label_metrics["Cardiomegaly"].auroc == 1.0
    assert report.aggregate_metrics.n_labels == 2


def test_duplicate_record_image_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate data record image_id"):
        build_evaluation_rows_from_records_and_predictions(
            records=[_record("img001"), _record("img001")],
            predictions=_batch([_prediction("img001")]),
            labels=LABELS,
        )


def test_duplicate_prediction_sample_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate prediction sample_id"):
        _batch([_prediction("img001"), _prediction("img001")])


def test_missing_prediction_for_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing predictions.*img002"):
        build_evaluation_rows_from_records_and_predictions(
            records=[_record("img001"), _record("img002")],
            predictions=_batch([_prediction("img001")]),
            labels=LABELS,
        )


def test_unknown_prediction_image_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown image_ids.*img002"):
        build_evaluation_rows_from_records_and_predictions(
            records=[_record("img001")],
            predictions=_batch([_prediction("img001"), _prediction("img002")]),
            labels=LABELS,
        )


def test_missing_requested_label_in_data_record_is_rejected() -> None:
    record = _record("img001")
    record.labels.pop("Cardiomegaly")

    with pytest.raises(ValueError, match="data record.*Cardiomegaly"):
        build_evaluation_rows_from_records_and_predictions(
            records=[record],
            predictions=_batch([_prediction("img001")]),
            labels=LABELS,
        )


def test_missing_requested_label_in_prediction_record_is_rejected() -> None:
    prediction = _prediction("img001", scores={"Atelectasis": 0.2})
    predictions = _batch([prediction], labels=("Atelectasis",))

    with pytest.raises(ValueError, match="prediction record.*Cardiomegaly"):
        build_evaluation_rows_from_records_and_predictions(
            records=[_record("img001")],
            predictions=predictions,
            labels=LABELS,
        )


def test_empty_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        build_evaluation_rows_from_records_and_predictions(
            records=[_record("img001")],
            predictions=_batch([_prediction("img001")]),
            labels=(),
        )
