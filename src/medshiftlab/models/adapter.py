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
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")

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

    @property
    def model_name(self) -> str:
        """Return the configured mock model name."""

        return self._model_name

    @property
    def labels(self) -> tuple[str, ...]:
        """Return configured labels in prediction order."""

        return self._labels

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        """Return constant deterministic scores without loading images."""

        records: list[PredictionRecord] = []
        for image_record in image_records:
            image_id = image_record.get("image_id")
            if not isinstance(image_id, str):
                raise ValueError("each image record must contain a string image_id")
            image_path = image_record.get("image_path")
            if image_path is not None and not isinstance(image_path, str):
                raise ValueError("image_path must be a string or None")
            dataset_name = image_record.get("dataset_name")
            if dataset_name is not None and not isinstance(dataset_name, str):
                raise ValueError("dataset_name must be a string or None")

            records.append(
                PredictionRecord(
                    image_id=image_id,
                    image_path=image_path,
                    dataset_name=dataset_name,
                    model_name=self.model_name,
                    scores={label: self._default_score for label in self.labels},
                )
            )

        return PredictionBatch(
            model_name=self.model_name,
            records=records,
            labels=self.labels,
        )
