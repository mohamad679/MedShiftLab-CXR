"""Focused Phase 9 tests for safe adapter selection and scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from medshiftlab.models import (
    CXRModelAdapter,
    FoundationModelAdapter,
    FoundationModelAdapterConfig,
    PredictionBatch,
    create_model_adapter,
    get_adapter_candidate,
    list_adapter_candidates,
    list_adapter_keys,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeFoundationBackend:
    def predict(self, image_records):
        return [[0.2, 0.8] for _ in image_records]


def _config(**overrides: object) -> FoundationModelAdapterConfig:
    values: dict[str, object] = {
        "model_name": "foundation-test-fixture",
        "checkpoint_source": "mock-source:no-weights",
        "labels": ("Atelectasis", "Cardiomegaly"),
    }
    values.update(overrides)
    return FoundationModelAdapterConfig(**values)


def test_registry_lists_expected_adapters_without_initialization() -> None:
    assert list_adapter_keys() == (
        "mock_constant",
        "torchxrayvision",
        "foundation_model_scaffold",
    )
    assert {candidate.implementation_status for candidate in list_adapter_candidates()} <= {
        "planned",
        "mock_only",
        "manual_optional",
        "implemented",
    }
    assert all(not candidate.automatic_weight_download for candidate in list_adapter_candidates())
    assert all(
        candidate.weight_download_policy == "forbidden_by_default"
        for candidate in list_adapter_candidates()
    )


def test_unknown_adapter_key_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown adapter key.*mock_constant"):
        get_adapter_candidate("not-registered")


def test_factory_creates_safe_mock_adapter_by_default() -> None:
    adapter = create_model_adapter(
        "mock_constant", model_name="fixture", labels=("Atelectasis",)
    )
    assert isinstance(adapter, CXRModelAdapter)


def test_factory_refuses_optional_real_adapter_by_default() -> None:
    with pytest.raises(PermissionError, match="disabled by default"):
        create_model_adapter("torchxrayvision")


def test_foundation_scaffold_returns_standardized_prediction_batch() -> None:
    adapter = create_model_adapter(
        "foundation_model_scaffold",
        config=_config(),
        mock_backend=_FakeFoundationBackend(),
    )
    batch = adapter.predict_records(
        [{"sample_id": "synthetic-001", "dataset_name": "synthetic"}]
    )

    assert isinstance(adapter, FoundationModelAdapter)
    assert isinstance(adapter, CXRModelAdapter)
    assert isinstance(batch, PredictionBatch)
    assert batch.records[0].probabilities == (0.2, 0.8)
    assert batch.run_metadata == {
        "backend_mode": "injected_mock",
        "real_inference": False,
        "weights_downloaded": False,
    }


def test_foundation_scaffold_preserves_preprocessing_provenance() -> None:
    config = _config(
        preprocessing_config={
            "image_loader": "phase3-package-loader",
            "resize": [224, 224],
            "weight_download": "forbidden",
        }
    )
    adapter = FoundationModelAdapter(config, mock_backend=_FakeFoundationBackend())

    batch = adapter.predict_records(
        [{"image_id": "synthetic-001", "dataset_name": "synthetic"}]
    )

    assert batch.preprocessing_version == "phase9-foundation-scaffold-v1"
    assert batch.preprocessing_config == config.preprocessing_config


def test_real_foundation_backend_is_refused_before_dependency_checks(monkeypatch) -> None:
    dependency_check_called = False

    def _unexpected_find_spec(_package: str):
        nonlocal dependency_check_called
        dependency_check_called = True
        raise AssertionError("dependency discovery must not run without authorization")

    monkeypatch.setattr(
        "medshiftlab.models.foundation_adapter.find_spec", _unexpected_find_spec
    )
    with pytest.raises(PermissionError, match="disabled by default"):
        FoundationModelAdapter(_config())
    assert dependency_check_called is False


def test_allowed_real_backend_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        "medshiftlab.models.foundation_adapter.find_spec", lambda _package: None
    )
    with pytest.raises(ImportError, match="torch, transformers.*no packages or weights"):
        FoundationModelAdapter(_config(), allow_real_backend=True)


def test_allowed_real_backend_remains_unimplemented_without_importing(monkeypatch) -> None:
    monkeypatch.setattr(
        "medshiftlab.models.foundation_adapter.find_spec", lambda _package: object()
    )
    with pytest.raises(NotImplementedError, match="not implemented in Phase 9"):
        FoundationModelAdapter(_config(), allow_real_backend=True)


def test_mock_output_and_safe_configs_contain_no_private_paths() -> None:
    adapter = FoundationModelAdapter(_config(), mock_backend=_FakeFoundationBackend())
    batch = adapter.predict_records(
        [{"sample_id": "synthetic-001", "dataset_name": "synthetic"}]
    )
    payloads = [batch.model_dump_json()]
    for relative_path in (
        "configs/models/adapter_candidates.yaml",
        "configs/experiments/manual_foundation_adapter.yaml",
    ):
        config_path = REPO_ROOT / relative_path
        payloads.append(config_path.read_text(encoding="utf-8"))
        assert isinstance(yaml.safe_load(payloads[-1]), dict)

    serialized = json.dumps(payloads)
    assert "/Users/" not in serialized
    assert "C:\\" not in serialized


def test_mock_backend_never_loads_images_or_downloads_weights() -> None:
    adapter = FoundationModelAdapter(_config(), mock_backend=_FakeFoundationBackend())
    batch = adapter.predict_records(
        [{"sample_id": "metadata-only", "dataset_name": "synthetic"}]
    )
    assert batch.records[0].image_path is None
    assert batch.preprocessing_config["real_image_loading"] is False
    assert batch.preprocessing_config["weight_download"] == "forbidden"
