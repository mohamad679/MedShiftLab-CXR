"""Optional TorchXRayVision adapter boundary without eager dependencies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from medshiftlab.data.image_loader import ImagePreprocessingConfig, load_dataset_image
from medshiftlab.data.registry import LocalDataConfig
from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


_DEPENDENCY_MESSAGE = (
    "TorchXRayVision support requires optional 'torch' and 'torchxrayvision' "
    "dependencies. Install them separately. Real inference also requires an "
    "initialized local model; this adapter does not download or manage weights."
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
        default="phase5-baseline-inference-v1",
        min_length=1,
    )
    preprocessing_config: dict[str, object] = Field(default_factory=dict)
    image_preprocessing: ImagePreprocessingConfig = Field(
        default_factory=lambda: ImagePreprocessingConfig(
            output_mode="grayscale",
            target_size=(224, 224),
            normalization="minus_one_one",
        )
    )
    batch_size: int = Field(default=4, gt=0)
    output_activation: Literal["identity", "sigmoid"] = "identity"

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
        *,
        local_data_config: LocalDataConfig | None = None,
    ) -> None:
        if model is None and not is_torchxrayvision_available():
            raise ImportError(_DEPENDENCY_MESSAGE)

        self.config = config
        self._model = model
        self._local_data_config = local_data_config

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
        """Return preprocessing provenance for baseline local inference."""

        metadata: dict[str, object] = {
            "image_loader": "phase3-package-loader",
            "image_preprocessing": self.config.image_preprocessing.model_dump(
                mode="json"
            ),
            "batch_size": self.config.batch_size,
            "output_activation": self.config.output_activation,
        }
        metadata.update(self.config.preprocessing_config)
        return metadata

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

        return self._build_prediction_batch(
            image_records=image_records,
            model_outputs=model_outputs,
            run_metadata={},
        )

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        """Run bounded local inference with an externally initialized model."""

        if not image_records:
            raise ValueError("image_records must not be empty")
        if self._model is None:
            raise RuntimeError(
                "predict_records requires an initialized local model. "
                "This adapter does not create or download weights."
            )
        if self._local_data_config is None:
            raise ValueError(
                "predict_records requires local_data_config for dataset-registry image "
                "resolution"
            )

        torch = _load_torch_dependency()
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.config.device}")

        if hasattr(self._model, "eval"):
            self._model.eval()

        output_rows: list[list[float]] = []
        for start in range(0, len(image_records), self.config.batch_size):
            batch_records = image_records[start : start + self.config.batch_size]
            batch_tensor = _build_model_input_tensor(
                torch=torch,
                image_records=batch_records,
                local_data_config=self._local_data_config,
                preprocessing=self.config.image_preprocessing,
            ).to(self.config.device)
            with torch.inference_mode():
                raw_outputs = self._model(batch_tensor)
            output_rows.extend(_coerce_model_output_rows(raw_outputs))

        if len(output_rows) != len(image_records):
            raise RuntimeError(
                "Model output row count does not match the number of requested records"
            )

        return self._build_prediction_batch(
            image_records=image_records,
            model_outputs=output_rows,
            run_metadata={
                "inference_mode": "baseline_local_subset",
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "n_records": len(image_records),
                "model_initialized_externally": True,
            },
        )

    def _build_prediction_batch(
        self,
        *,
        image_records: Sequence[Mapping[str, object]],
        model_outputs: Sequence[_OutputRow],
        run_metadata: Mapping[str, object],
    ) -> PredictionBatch:
        predictions: list[PredictionRecord] = []
        for row_number, (image_record, output_row) in enumerate(
            zip(image_records, model_outputs, strict=True)
        ):
            sample_id = _required_sample_id(image_record, row_number)
            image_path = _optional_string(image_record, "image_path", row_number)
            dataset_name = _required_dataset_name(image_record, row_number)
            patient_id = _optional_string(image_record, "patient_id", row_number)
            study_id = _optional_string(image_record, "study_id", row_number)

            probabilities: list[float] = []
            logits: list[float] = []
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

                if self.config.output_activation == "sigmoid":
                    logits.append(score)
                    probability = _sigmoid(score)
                else:
                    probability = score
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"output score for {label!r} in row {row_number} "
                        "must produce a probability between 0 and 1"
                    )
                probabilities.append(probability)

            predictions.append(
                PredictionRecord(
                    sample_id=sample_id,
                    image_path=image_path,
                    dataset_name=dataset_name,
                    model_name=self.model_name,
                    patient_id=patient_id,
                    study_id=study_id,
                    label_names=self.labels,
                    probabilities=tuple(probabilities),
                    logits=tuple(logits) if logits else None,
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
            run_metadata=dict(run_metadata),
        )


def _build_model_input_tensor(
    *,
    torch: Any,
    image_records: Sequence[Mapping[str, object]],
    local_data_config: LocalDataConfig,
    preprocessing: ImagePreprocessingConfig,
) -> Any:
    tensors: list[Any] = []
    for row_number, image_record in enumerate(image_records):
        sample_id = _required_sample_id(image_record, row_number)
        dataset_name = _required_dataset_name(image_record, row_number)
        image_path = _required_image_path(image_record, row_number)
        loaded = load_dataset_image(
            config=local_data_config,
            dataset_name=dataset_name,
            image_path=image_path,
            preprocessing=preprocessing,
        )
        tensors.append(_array_to_tensor(torch, loaded.array, sample_id))
    return torch.stack(tensors)


def _array_to_tensor(torch: Any, array: object, sample_id: str) -> Any:
    tensor = torch.as_tensor(array)
    shape = getattr(tensor, "shape", ())
    if len(shape) == 2:
        return tensor.unsqueeze(0)
    if len(shape) == 3:
        if shape[-1] in (1, 3):
            return tensor.permute(2, 0, 1)
        if shape[0] in (1, 3):
            return tensor
    raise ValueError(
        f"Unsupported preprocessed image shape for sample {sample_id!r}: {tuple(shape)}"
    )


def _coerce_model_output_rows(raw_outputs: object) -> list[list[float]]:
    candidate = raw_outputs
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()

    if not isinstance(candidate, Sequence):
        raise RuntimeError("Model outputs must be a sequence of rows")

    rows: list[list[float]] = []
    for row in candidate:
        if not isinstance(row, Sequence):
            raise RuntimeError("Each model output row must be a sequence")
        rows.append([float(value) for value in row])
    return rows


def _load_torch_dependency() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError(_DEPENDENCY_MESSAGE) from error


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _required_sample_id(image_record: Mapping[str, object], row_number: int) -> str:
    sample_id = image_record.get("sample_id", image_record.get("image_id"))
    if not isinstance(sample_id, str):
        raise ValueError(
            f"image record {row_number} must contain a string sample_id or image_id"
        )
    return sample_id.strip()


def _required_image_path(image_record: Mapping[str, object], row_number: int) -> str:
    image_path = image_record.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(
            f"image record {row_number} must contain a non-empty string image_path"
        )
    return image_path.strip()


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
    return value.strip() if isinstance(value, str) and value.strip() else None
