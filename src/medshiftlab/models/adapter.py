"""Stable model adapter protocol and deterministic test adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


@runtime_checkable
class CXRModelAdapter(Protocol):
    """Interface implemented by pretrained chest X-ray model adapters."""

    @property
    def model_name(self) -> str:
        """Return the stable model identifier."""
        ...

    @property
    def labels(self) -> tuple[str, ...]:
        """Return prediction labels in stable output order."""
        ...

    @property
    def model_version(self) -> str:
        """Return the model version or checkpoint/source identifier."""
        ...

    @property
    def adapter_name(self) -> str:
        """Return the stable adapter implementation identifier."""
        ...

    @property
    def preprocessing_version(self) -> str:
        """Return the preprocessing contract version for this adapter."""
        ...

    @property
    def preprocessing_config(self) -> Mapping[str, object]:
        """Return model-independent preprocessing provenance."""
        ...

    @property
    def uncertainty_strategy(self) -> str | None:
        """Return the adapter-level uncertainty strategy if fixed."""
        ...

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        """Return predictions for image-record metadata."""
        ...


class MockCXRModelAdapter:
    """Deterministic constant-score adapter for tests and pipeline fixtures."""

    def __init__(
        self,
        model_name: str,
        labels: Sequence[str],
        default_score: float = 0.5,
        *,
        model_version: str = "mock-constant-v1",
        adapter_name: str = "mock-cxr-adapter",
        preprocessing_version: str = "mock-preprocessing-v1",
        preprocessing_config: Mapping[str, object] | None = None,
        uncertainty_strategy: str | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        model_version = model_version.strip()
        if not model_version:
            raise ValueError("model_version must not be empty")
        adapter_name = adapter_name.strip()
        if not adapter_name:
            raise ValueError("adapter_name must not be empty")
        preprocessing_version = preprocessing_version.strip()
        if not preprocessing_version:
            raise ValueError("preprocessing_version must not be empty")

        label_tuple = tuple(labels)
        if not label_tuple:
            raise ValueError("labels must not be empty")
        if any(not label.strip() for label in label_tuple):
            raise ValueError("labels must contain only non-empty strings")
        if not 0.0 <= default_score <= 1.0:
            raise ValueError("default_score must be between 0 and 1")

        self._model_name = model_name
        self._labels = label_tuple
        self._default_score = float(default_score)
        self._model_version = model_version
        self._adapter_name = adapter_name
        self._preprocessing_version = preprocessing_version
        self._preprocessing_config = dict(
            preprocessing_config
            or {
                "kind": "synthetic-constant-score",
                "real_image_loading": False,
                "default_score": self._default_score,
            }
        )
        self._uncertainty_strategy = (
            uncertainty_strategy.strip() if uncertainty_strategy else None
        )

    @property
    def model_name(self) -> str:
        """Return the configured mock model name."""

        return self._model_name

    @property
    def labels(self) -> tuple[str, ...]:
        """Return configured labels in prediction order."""

        return self._labels

    @property
    def model_version(self) -> str:
        """Return the configured mock model version."""

        return self._model_version

    @property
    def adapter_name(self) -> str:
        """Return the configured mock adapter name."""

        return self._adapter_name

    @property
    def preprocessing_version(self) -> str:
        """Return the configured preprocessing version."""

        return self._preprocessing_version

    @property
    def preprocessing_config(self) -> Mapping[str, object]:
        """Return mock preprocessing provenance."""

        return self._preprocessing_config

    @property
    def uncertainty_strategy(self) -> str | None:
        """Return the configured uncertainty strategy, if any."""

        return self._uncertainty_strategy

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        """Return constant deterministic scores without loading images."""

        records: list[PredictionRecord] = []
        for image_record in image_records:
            sample_id = image_record.get("sample_id", image_record.get("image_id"))
            if not isinstance(sample_id, str):
                raise ValueError(
                    "each image record must contain a string sample_id or image_id"
                )
            image_path = image_record.get("image_path")
            if image_path is not None and not isinstance(image_path, str):
                raise ValueError("image_path must be a string or None")
            dataset_name = image_record.get("dataset_name")
            if not isinstance(dataset_name, str) or not dataset_name.strip():
                raise ValueError(
                    "each image record must contain a non-empty string dataset_name"
                )
            patient_id = image_record.get("patient_id")
            if patient_id is not None and not isinstance(patient_id, str):
                raise ValueError("patient_id must be a string or None")
            study_id = image_record.get("study_id")
            if study_id is not None and not isinstance(study_id, str):
                raise ValueError("study_id must be a string or None")

            records.append(
                PredictionRecord(
                    sample_id=sample_id,
                    image_path=image_path,
                    dataset_name=dataset_name,
                    model_name=self.model_name,
                    patient_id=patient_id,
                    study_id=study_id,
                    label_names=self.labels,
                    probabilities=tuple(
                        self._default_score for _ in self.labels
                    ),
                )
            )

        return PredictionBatch(
            model_name=self.model_name,
            model_version=self.model_version,
            adapter_name=self.adapter_name,
            preprocessing_version=self.preprocessing_version,
            preprocessing_config=dict(self.preprocessing_config),
            records=records,
            label_names=self.labels,
            uncertainty_strategy=self.uncertainty_strategy,
        )
