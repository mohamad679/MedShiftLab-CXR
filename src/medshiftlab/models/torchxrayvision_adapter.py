"""Optional TorchXRayVision adapter boundary without eager dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


_DEPENDENCY_MESSAGE = (
    "TorchXRayVision support requires optional 'torch' and 'torchxrayvision' "
    "dependencies; install them separately and provide an initialized model."
)


class _OutputRow(Protocol):
    def __getitem__(self, index: int) -> object: ...


def is_torchxrayvision_available() -> bool:
    """Return whether both optional inference packages can be discovered."""

    try:
        return (
            find_spec("torch") is not None
            and find_spec("torchxrayvision") is not None
        )
    except (ImportError, ValueError):
        return False


class TorchXRayVisionAdapterConfig(BaseModel):
    """Configuration for mapping model output columns to project labels."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)
    output_indices: dict[str, StrictInt]
    device: str = Field(default="cpu", min_length=1)
    adapter_name: str = Field(default="torchxrayvision-adapter", min_length=1)
    preprocessing_version: str = Field(
        default="torchxrayvision-preprocessing-v1",
        min_length=1,
    )
    preprocessing_config: dict[str, object] = Field(
        default_factory=lambda: {
            "image_loader": "phase3-package-loader",
            "inference_integration": "not-yet-implemented",
        }
    )

    @field_validator(
        "model_name",
        "model_version",
        "device",
        "adapter_name",
        "preprocessing_version",
    )
    @classmethod
    def _strip_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(
                "model_name, model_version, device, adapter_name, and "
                "preprocessing_version must not be empty"
            )
        return value

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if any(not label.strip() for label in labels):
            raise ValueError("labels must contain only non-empty strings")
        return labels

    @field_validator("output_indices")
    @classmethod
    def _validate_output_indices(
        cls, output_indices: dict[str, int]
    ) -> dict[str, int]:
        for label, index in output_indices.items():
            if not label.strip():
                raise ValueError("output index labels must not be empty")
            if index < 0:
                raise ValueError("output indices must be non-negative integers")
        return output_indices

    @model_validator(mode="after")
    def _validate_requested_output_indices(self) -> TorchXRayVisionAdapterConfig:
        missing = [label for label in self.labels if label not in self.output_indices]
        if missing:
            raise ValueError(
                "output_indices is missing requested labels: " + ", ".join(missing)
            )
        return self


class TorchXRayVisionAdapter:
    """Dependency-safe boundary for externally initialized TorchXRayVision models."""

    def __init__(
        self,
        config: TorchXRayVisionAdapterConfig,
        model: object | None = None,
    ) -> None:
        if model is None and not is_torchxrayvision_available():
            raise ImportError(_DEPENDENCY_MESSAGE)

        self.config = config
        self._model = model

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

        return self.config.model_name

    @property
    def labels(self) -> tuple[str, ...]:
        """Return configured project labels in output order."""

        return self.config.labels

    @property
    def model_version(self) -> str:
        """Return the configured model version or source identifier."""

        return self.config.model_version

    @property
    def adapter_name(self) -> str:
        """Return the configured adapter implementation identifier."""

        return self.config.adapter_name

    @property
    def preprocessing_version(self) -> str:
        """Return the configured preprocessing contract version."""

        return self.config.preprocessing_version

    @property
    def preprocessing_config(self) -> Mapping[str, object]:
        """Return preprocessing provenance for future real-inference wiring."""

        return self.config.preprocessing_config

    @property
    def uncertainty_strategy(self) -> str | None:
        """Return a fixed uncertainty strategy if this adapter carries one."""

        return None

    def predict_scores_from_outputs(
        self,
        image_records: Sequence[Mapping[str, object]],
        model_outputs: Sequence[_OutputRow],
    ) -> PredictionBatch:
        """Map precomputed output rows to validated project-label predictions."""

        if len(image_records) != len(model_outputs):
            raise ValueError("image_records and model_outputs must have equal lengths")

        predictions: list[PredictionRecord] = []
        for row_number, (image_record, output_row) in enumerate(
            zip(image_records, model_outputs, strict=True)
        ):
            image_id = _required_image_id(image_record, row_number)
            image_path = _optional_string(image_record, "image_path", row_number)
            dataset_name = _required_dataset_name(image_record, row_number)

            scores: dict[str, float] = {}
            for label in self.labels:
                output_index = self.config.output_indices[label]
                try:
                    raw_score = output_row[output_index]
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"output row {row_number} has no value at index {output_index}"
                    ) from exc
                try:
                    score = float(raw_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"output score for {label!r} in row {row_number} is not numeric"
                    ) from exc
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"output score for {label!r} in row {row_number} "
                        "must be between 0 and 1"
                    )
                scores[label] = score

            predictions.append(
                PredictionRecord(
                    image_id=image_id,
                    image_path=image_path,
                    dataset_name=dataset_name,
                    model_name=self.model_name,
                    label_names=self.labels,
                    probabilities=tuple(scores[label] for label in self.labels),
                )
            )

        return PredictionBatch(
            model_name=self.model_name,
            model_version=self.model_version,
            adapter_name=self.adapter_name,
            preprocessing_version=self.preprocessing_version,
            preprocessing_config=dict(self.preprocessing_config),
            records=predictions,
            label_names=self.labels,
            uncertainty_strategy=self.uncertainty_strategy,
        )

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        """Reject real inference until preprocessing is implemented explicitly."""

        raise NotImplementedError(
            "Real image preprocessing and TorchXRayVision inference are intentionally "
            "not implemented in this adapter boundary."
        )


def _required_image_id(image_record: Mapping[str, object], row_number: int) -> str:
    image_id = image_record.get("image_id")
    if not isinstance(image_id, str):
        raise ValueError(f"image record {row_number} must contain a string image_id")
    return image_id


def _required_dataset_name(
    image_record: Mapping[str, object], row_number: int
) -> str:
    dataset_name = image_record.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(
            f"image record {row_number} must contain a non-empty string dataset_name"
        )
    return dataset_name.strip()


def _optional_string(
    image_record: Mapping[str, object], field_name: str, row_number: int
) -> str | None:
    value = image_record.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} in image record {row_number} must be a string")
    return value
