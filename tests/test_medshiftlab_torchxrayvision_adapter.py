"""Tests for the dependency-safe optional TorchXRayVision adapter."""

from __future__ import annotations

import pytest

import medshiftlab.models as models
from medshiftlab.models import (
    PredictionBatch,
    TorchXRayVisionAdapter,
    TorchXRayVisionAdapterConfig,
    is_torchxrayvision_available,
)


def _config() -> TorchXRayVisionAdapterConfig:
    return TorchXRayVisionAdapterConfig(
        model_name="torchxrayvision-densenet121",
        model_version="torchxrayvision-densenet121:external-init",
        labels=("Atelectasis", "Cardiomegaly"),
        output_indices={"Atelectasis": 1, "Cardiomegaly": 0},
    )


def _adapter() -> TorchXRayVisionAdapter:
    return TorchXRayVisionAdapter(config=_config(), model=object())


def test_models_package_imports_without_optional_dependency() -> None:
    assert models.TorchXRayVisionAdapter is TorchXRayVisionAdapter


def test_is_torchxrayvision_available_returns_bool() -> None:
    assert isinstance(is_torchxrayvision_available(), bool)


def test_adapter_without_model_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        "medshiftlab.models.torchxrayvision_adapter.is_torchxrayvision_available",
        lambda: False,
    )

    with pytest.raises(ImportError, match="optional 'torch' and 'torchxrayvision'"):
        TorchXRayVisionAdapter(config=_config())


def test_config_rejects_empty_labels() -> None:
    with pytest.raises(ValueError):
        TorchXRayVisionAdapterConfig(
            model_name="model-a",
            model_version="model-a:v1",
            labels=(),
            output_indices={},
        )


def test_config_rejects_missing_output_index() -> None:
    with pytest.raises(ValueError, match="Cardiomegaly"):
        TorchXRayVisionAdapterConfig(
            model_name="model-a",
            model_version="model-a:v1",
            labels=("Atelectasis", "Cardiomegaly"),
            output_indices={"Atelectasis": 0},
        )


def test_config_rejects_negative_output_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TorchXRayVisionAdapterConfig(
            model_name="model-a",
            model_version="model-a:v1",
            labels=("Atelectasis",),
            output_indices={"Atelectasis": -1},
        )


def test_config_rejects_non_integer_output_index() -> None:
    with pytest.raises(ValueError):
        TorchXRayVisionAdapterConfig(
            model_name="model-a",
            model_version="model-a:v1",
            labels=("Atelectasis",),
            output_indices={"Atelectasis": 0.5},
        )


def test_predict_scores_from_outputs_maps_to_prediction_batch() -> None:
    batch = _adapter().predict_scores_from_outputs(
        image_records=[
            {"image_id": "img001", "dataset_name": "CheXpert"},
            {"image_id": "img002", "dataset_name": "CheXpert"},
        ],
        model_outputs=[[0.8, 0.2], [0.3, 0.7]],
    )

    assert isinstance(batch, PredictionBatch)
    assert batch.model_name == "torchxrayvision-densenet121"
    assert batch.model_version == "torchxrayvision-densenet121:external-init"
    assert batch.adapter_name == "torchxrayvision-adapter"
    assert batch.preprocessing_version == "torchxrayvision-preprocessing-v1"
    assert batch.labels == ("Atelectasis", "Cardiomegaly")
    assert [record.scores for record in batch.records] == [
        {"Atelectasis": 0.2, "Cardiomegaly": 0.8},
        {"Atelectasis": 0.7, "Cardiomegaly": 0.3},
    ]


def test_predict_scores_from_outputs_preserves_image_metadata() -> None:
    batch = _adapter().predict_scores_from_outputs(
        image_records=[
            {
                "image_id": "img001",
                "image_path": "images/img001.png",
                "dataset_name": "CheXpert",
            }
        ],
        model_outputs=[[0.8, 0.2]],
    )

    record = batch.records[0]
    assert record.sample_id == "img001"
    assert record.image_id == "img001"
    assert record.image_path == "images/img001.png"
    assert record.dataset_name == "CheXpert"


def test_predict_scores_from_outputs_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        _adapter().predict_scores_from_outputs(
            image_records=[{"image_id": "img001", "dataset_name": "CheXpert"}],
            model_outputs=[],
        )


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1])
def test_predict_scores_from_outputs_rejects_invalid_probability(
    invalid_score: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _adapter().predict_scores_from_outputs(
            image_records=[{"image_id": "img001", "dataset_name": "CheXpert"}],
            model_outputs=[[0.8, invalid_score]],
        )


def test_predict_scores_from_outputs_rejects_missing_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        _adapter().predict_scores_from_outputs(
            image_records=[{"image_id": "img001"}],
            model_outputs=[[0.8, 0.2]],
        )


def test_predict_records_explicitly_rejects_real_inference() -> None:
    with pytest.raises(NotImplementedError, match="intentionally not implemented"):
        _adapter().predict_records([{"image_id": "img001"}])
