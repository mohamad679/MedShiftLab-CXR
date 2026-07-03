"""Tests for the MedShiftLab-CXR in-memory experiment runner."""

from __future__ import annotations

import pytest

from medshiftlab.data import CheXpertRecord
from medshiftlab.evaluation import EvaluationReport
from medshiftlab.experiments import (
    InMemoryExperimentConfig,
    InMemoryExperimentResult,
    run_in_memory_evaluation_experiment,
)
from medshiftlab.models import MockCXRModelAdapter, PredictionBatch


LABELS = ("Atelectasis", "Cardiomegaly")


def _config() -> InMemoryExperimentConfig:
    return InMemoryExperimentConfig(
        dataset_name="CheXpert",
        labels=LABELS,
        split="validation",
        uncertainty_strategy="U-ignore",
        threshold=0.5,
        n_bins=2,
        notes="in-memory fixture",
    )


def _record(
    image_id: str, atelectasis: float = 0.0, cardiomegaly: float = 1.0
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


def _adapter() -> MockCXRModelAdapter:
    return MockCXRModelAdapter("mock-cxr", LABELS, default_score=0.25)


def test_runner_returns_in_memory_experiment_result() -> None:
    result = run_in_memory_evaluation_experiment(
        [_record("img001")], _adapter(), _config()
    )

    assert isinstance(result, InMemoryExperimentResult)
    assert result.n_records == 1
    assert result.model_name == "mock-cxr"


def test_runner_produces_prediction_batch_from_mock_adapter() -> None:
    result = run_in_memory_evaluation_experiment(
        [_record("img001"), _record("img002")], _adapter(), _config()
    )

    assert isinstance(result.predictions, PredictionBatch)
    assert [record.scores for record in result.predictions.records] == [
        {"Atelectasis": 0.25, "Cardiomegaly": 0.25},
        {"Atelectasis": 0.25, "Cardiomegaly": 0.25},
    ]


def test_runner_produces_report_with_experiment_metadata() -> None:
    result = run_in_memory_evaluation_experiment(
        [_record("img001"), _record("img002", 1.0, 0.0)],
        _adapter(),
        _config(),
    )

    assert isinstance(result.report, EvaluationReport)
    assert result.report.metadata.dataset_name == "CheXpert"
    assert result.report.metadata.model_name == "mock-cxr"
    assert result.report.metadata.split == "validation"
    assert result.report.metadata.uncertainty_strategy == "U-ignore"
    assert result.report.metadata.notes == "in-memory fixture"


def test_runner_produces_evaluation_table_columns() -> None:
    result = run_in_memory_evaluation_experiment(
        [_record("img001")], _adapter(), _config()
    )

    assert result.evaluation_rows == [
        {
            "image_id": "img001",
            "image_path": "images/img001.png",
            "dataset_name": "CheXpert",
            "model_name": "mock-cxr",
            "true_Atelectasis": 0.0,
            "score_Atelectasis": 0.25,
            "true_Cardiomegaly": 1.0,
            "score_Cardiomegaly": 0.25,
        }
    ]


def test_runner_preserves_input_record_order() -> None:
    result = run_in_memory_evaluation_experiment(
        [_record("img002"), _record("img001")], _adapter(), _config()
    )

    assert [row["image_id"] for row in result.evaluation_rows] == [
        "img002",
        "img001",
    ]
    assert [record.image_id for record in result.predictions.records] == [
        "img002",
        "img001",
    ]


def test_runner_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="records must not be empty"):
        run_in_memory_evaluation_experiment([], _adapter(), _config())


def test_config_rejects_empty_labels() -> None:
    with pytest.raises(ValueError):
        InMemoryExperimentConfig(dataset_name="CheXpert", labels=())


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_config_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        InMemoryExperimentConfig(
            dataset_name="CheXpert", labels=LABELS, threshold=threshold
        )


@pytest.mark.parametrize("n_bins", [0, -1])
def test_config_rejects_invalid_bin_count(n_bins: int) -> None:
    with pytest.raises(ValueError):
        InMemoryExperimentConfig(
            dataset_name="CheXpert", labels=LABELS, n_bins=n_bins
        )


def test_runner_accepts_mapping_records() -> None:
    records = [
        {
            "image_id": "img001",
            "image_path": "images/img001.png",
            "dataset_name": "CheXpert",
            "labels": {"Atelectasis": 0.0, "Cardiomegaly": 1.0},
        }
    ]

    result = run_in_memory_evaluation_experiment(records, _adapter(), _config())

    assert result.evaluation_rows[0]["image_id"] == "img001"
    assert result.evaluation_rows[0]["true_Atelectasis"] == 0.0
    assert result.report.metadata.dataset_name == "CheXpert"
