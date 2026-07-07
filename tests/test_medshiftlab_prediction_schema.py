"""Focused tests for the standardized prediction schema and exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from medshiftlab.models import (
    MockCXRModelAdapter,
    PREDICTION_SCHEMA_VERSION,
    PredictionBatch,
    PredictionRecord,
)
from medshiftlab.reporting import (
    read_prediction_batch_json,
    read_prediction_records_csv,
    write_prediction_batch_json,
    write_prediction_records_csv,
)


def _prediction_record(**overrides: object) -> PredictionRecord:
    payload: dict[str, object] = {
        "sample_id": "img001",
        "patient_id": "patient-1",
        "study_id": "study-1",
        "dataset_name": "CheXpert",
        "image_path": "images/img001.png",
        "model_name": "mock-cxr",
        "label_names": ("Atelectasis", "Cardiomegaly"),
        "probabilities": (0.2, 0.8),
    }
    payload.update(overrides)
    return PredictionRecord(**payload)


def _prediction_batch(
    records: list[PredictionRecord] | None = None,
    **overrides: object,
) -> PredictionBatch:
    payload: dict[str, object] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "model_name": "mock-cxr",
        "model_version": "mock-cxr:v1",
        "adapter_name": "mock-cxr-adapter",
        "preprocessing_version": "preprocess-v1",
        "preprocessing_config": {"output_mode": "grayscale", "target_size": [224, 224]},
        "label_names": ("Atelectasis", "Cardiomegaly"),
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "records": records or [_prediction_record()],
    }
    payload.update(overrides)
    return PredictionBatch(**payload)


def test_valid_prediction_record_passes() -> None:
    record = _prediction_record(
        logits=(-1.2, 1.5),
        thresholds=0.5,
        thresholded_predictions=(0, 1),
    )

    assert record.sample_id == "img001"
    assert record.image_id == "img001"
    assert record.scores == {"Atelectasis": 0.2, "Cardiomegaly": 0.8}
    assert record.threshold_mapping == {"Atelectasis": 0.5, "Cardiomegaly": 0.5}


def test_probability_label_length_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="probabilities must align"):
        _prediction_record(probabilities=(0.2,))


def test_logits_label_length_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="logits must align"):
        _prediction_record(logits=(0.1,))


def test_thresholded_predictions_without_thresholds_fail() -> None:
    with pytest.raises(ValueError, match="thresholded_predictions must not exist"):
        _prediction_record(thresholded_predictions=(0, 1))


def test_threshold_shape_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="thresholds must align"):
        _prediction_record(thresholds=(0.4,))


def test_batch_requires_model_and_preprocessing_provenance() -> None:
    with pytest.raises(ValueError, match="preprocessing_config"):
        _prediction_batch(preprocessing_config={})

    with pytest.raises(ValueError, match="model_version"):
        _prediction_batch(model_version=" ")


def test_adapter_output_conforms_to_schema() -> None:
    adapter = MockCXRModelAdapter(
        model_name="mock-cxr",
        labels=("Atelectasis", "Cardiomegaly"),
        default_score=0.25,
        preprocessing_config={"output_mode": "grayscale", "normalization": "zero_one"},
    )

    batch = adapter.predict_records(
        [
            {
                "image_id": "img001",
                "dataset_name": "CheXpert",
                "image_path": "images/img001.png",
            }
        ]
    )

    assert batch.schema_version == PREDICTION_SCHEMA_VERSION
    assert batch.model_version == "mock-constant-v1"
    assert batch.adapter_name == "mock-cxr-adapter"
    assert batch.preprocessing_version == "mock-preprocessing-v1"
    assert batch.records[0].label_names == ("Atelectasis", "Cardiomegaly")
    assert batch.records[0].probabilities == (0.25, 0.25)


def test_json_export_import_round_trip_preserves_schema(tmp_path: Path) -> None:
    batch = _prediction_batch(
        records=[
            _prediction_record(
                logits=(-1.2, 1.5),
                thresholds=(0.4, 0.6),
                thresholded_predictions=(0, 1),
            )
        ]
    )

    output_path = write_prediction_batch_json(batch, tmp_path / "predictions.json")
    loaded = read_prediction_batch_json(output_path)

    assert loaded == batch


def test_csv_export_import_round_trip_preserves_schema(tmp_path: Path) -> None:
    batch = _prediction_batch(
        records=[
            _prediction_record(
                sample_id="img001",
                logits=(-1.2, 1.5),
                thresholds=0.5,
                thresholded_predictions=(0, 1),
            ),
            _prediction_record(
                sample_id="img002",
                patient_id=None,
                study_id=None,
                image_path="images/img002.png",
                probabilities=(0.6, 0.4),
                logits=(0.4, -0.2),
                thresholds=0.5,
                thresholded_predictions=(1, 0),
            ),
        ]
    )

    output_path = write_prediction_records_csv(batch, tmp_path / "predictions.csv")
    loaded = read_prediction_records_csv(output_path)

    assert loaded == batch


def test_prediction_exports_do_not_include_private_absolute_paths(
    tmp_path: Path,
) -> None:
    batch = _prediction_batch()

    json_path = write_prediction_batch_json(batch, tmp_path / "predictions.json")
    csv_path = write_prediction_records_csv(batch, tmp_path / "predictions.csv")

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))

    assert json_payload["records"][0]["image_path"] == "images/img001.png"
    assert csv_rows[0]["image_path"] == "images/img001.png"
    assert "/Users/mohsenshamsijazeb" not in json.dumps(json_payload)
    assert "/Users/mohsenshamsijazeb" not in csv_path.read_text(encoding="utf-8")
