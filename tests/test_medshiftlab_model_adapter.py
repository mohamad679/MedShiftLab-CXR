"""Tests for MedShiftLab-CXR prediction schemas and model adapters."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from medshiftlab.models import (
    CXRModelAdapter,
    MockCXRModelAdapter,
    PREDICTION_SCHEMA_VERSION,
    PredictionBatch,
    PredictionRecord,
    build_prediction_table_rows,
    build_score_mapping_from_predictions,
)


def _batch_kwargs() -> dict[str, object]:
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "model_name": "model-a",
        "model_version": "model-a-v1",
        "adapter_name": "mock-adapter",
        "preprocessing_version": "preprocess-v1",
        "preprocessing_config": {"output_mode": "grayscale"},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_prediction_record_validates_probability_range(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        PredictionRecord(
            image_id="img001",
            model_name="mock-model",
            dataset_name="CheXpert",
            scores={"Atelectasis": score},
        )


@pytest.mark.parametrize(
    ("image_id", "model_name"),
    [("", "mock-model"), ("img001", ""), ("   ", "mock-model")],
)
def test_prediction_record_rejects_empty_identifiers(
    image_id: str, model_name: str
) -> None:
    with pytest.raises(ValueError):
        PredictionRecord(
            image_id=image_id,
            model_name=model_name,
            dataset_name="CheXpert",
            scores={"Atelectasis": 0.5},
        )


def test_prediction_batch_rejects_mixed_model_names() -> None:
    records = [
        PredictionRecord(
            image_id="img001",
            model_name="model-a",
            dataset_name="CheXpert",
            scores={"Atelectasis": 0.2},
        ),
        PredictionRecord(
            image_id="img002",
            model_name="model-b",
            dataset_name="CheXpert",
            scores={"Atelectasis": 0.8},
        ),
    ]

    with pytest.raises(ValueError, match="batch model_name"):
        PredictionBatch(
            **_batch_kwargs(),
            records=records,
            label_names=("Atelectasis",),
        )


def test_prediction_batch_rejects_missing_requested_label() -> None:
    record = PredictionRecord(
        image_id="img001",
        model_name="model-a",
        dataset_name="CheXpert",
        scores={"Atelectasis": 0.2},
    )

    with pytest.raises(ValueError, match="Cardiomegaly"):
        PredictionBatch(
            **_batch_kwargs(),
            records=[record],
            label_names=("Atelectasis", "Cardiomegaly"),
        )


def test_build_score_mapping_preserves_label_and_prediction_order() -> None:
    predictions = [
        PredictionRecord(
            image_id="img002",
            model_name="model-a",
            dataset_name="CheXpert",
            scores={"Atelectasis": 0.2, "Cardiomegaly": 0.7},
        ),
        PredictionRecord(
            image_id="img001",
            model_name="model-a",
            dataset_name="CheXpert",
            scores={"Atelectasis": 0.8, "Cardiomegaly": None},
        ),
    ]

    scores = build_score_mapping_from_predictions(
        predictions, ["Cardiomegaly", "Atelectasis"]
    )

    assert tuple(scores) == ("Cardiomegaly", "Atelectasis")
    assert scores == {
        "Cardiomegaly": [0.7, None],
        "Atelectasis": [0.2, 0.8],
    }


def test_build_prediction_table_rows_creates_score_columns() -> None:
    predictions = [
        PredictionRecord(
            image_id="img001",
            image_path="images/img001.png",
            dataset_name="CheXpert",
            model_name="model-a",
            scores={"Atelectasis": 0.2, "Cardiomegaly": 0.7},
        )
    ]

    rows = build_prediction_table_rows(
        predictions, ["Atelectasis", "Cardiomegaly"]
    )

    assert rows == [
        {
            "sample_id": "img001",
            "image_id": "img001",
            "image_path": "images/img001.png",
            "dataset_name": "CheXpert",
            "model_name": "model-a",
            "score_Atelectasis": 0.2,
            "score_Cardiomegaly": 0.7,
        }
    ]


def test_mock_adapter_returns_deterministic_prediction_batch() -> None:
    adapter = MockCXRModelAdapter(
        model_name="mock-cxr",
        labels=("Atelectasis", "Cardiomegaly"),
        default_score=0.25,
    )
    image_records = [
        {"image_id": "img001", "dataset_name": "CheXpert"},
        {"image_id": "img002", "dataset_name": "CheXpert"},
    ]

    first = adapter.predict_records(image_records)
    second = adapter.predict_records(image_records)

    assert isinstance(adapter, CXRModelAdapter)
    assert isinstance(first, PredictionBatch)
    assert first.model_dump(exclude={"created_at"}) == second.model_dump(
        exclude={"created_at"}
    )
    assert first.created_at <= second.created_at
    assert first.schema_version == PREDICTION_SCHEMA_VERSION
    assert first.model_name == "mock-cxr"
    assert first.model_version == "mock-constant-v1"
    assert first.adapter_name == "mock-cxr-adapter"
    assert first.preprocessing_version == "mock-preprocessing-v1"
    assert first.preprocessing_config["real_image_loading"] is False
    assert first.labels == ("Atelectasis", "Cardiomegaly")
    assert [record.scores for record in first.records] == [
        {"Atelectasis": 0.25, "Cardiomegaly": 0.25},
        {"Atelectasis": 0.25, "Cardiomegaly": 0.25},
    ]


def test_mock_adapter_preserves_image_record_identity() -> None:
    adapter = MockCXRModelAdapter("mock-cxr", ("Atelectasis",))

    batch = adapter.predict_records(
        [
            {
                "image_id": "img001",
                "image_path": "images/img001.png",
                "dataset_name": "VinDr-CXR",
            }
        ]
    )

    record = batch.records[0]
    assert record.sample_id == "img001"
    assert record.image_id == "img001"
    assert record.image_path == "images/img001.png"
    assert record.dataset_name == "VinDr-CXR"


def test_prediction_batch_requires_non_empty_preprocessing_config() -> None:
    record = PredictionRecord(
        image_id="img001",
        model_name="model-a",
        dataset_name="CheXpert",
        scores={"Atelectasis": 0.2},
    )

    with pytest.raises(ValueError, match="preprocessing_config"):
        PredictionBatch(
            **{
                **_batch_kwargs(),
                "preprocessing_config": {},
                "records": [record],
                "label_names": ("Atelectasis",),
            }
        )
