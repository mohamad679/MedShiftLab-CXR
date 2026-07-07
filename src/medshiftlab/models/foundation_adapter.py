"""Mock-compatible foundation-model adapter scaffold without weight loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


class FoundationMockBackend(Protocol):
    """Minimal fake-backend contract used by Phase 9 tests and fixtures."""

    def predict(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> Sequence[Sequence[float]]: ...


class FoundationModelAdapterConfig(BaseModel):
    """Metadata and output contract for a future foundation-model integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    checkpoint_source: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)
    optional_dependencies: tuple[str, ...] = ("torch", "transformers")
    adapter_name: str = "foundation-model-scaffold"
    preprocessing_version: str = "phase9-foundation-scaffold-v1"
    preprocessing_config: dict[str, object] = Field(
        default_factory=lambda: {
            "image_loader": "phase3-package-loader",
            "backend_mode": "injected-mock-only",
            "real_image_loading": False,
            "weight_download": "forbidden",
        }
    )

    @field_validator(
        "model_name", "checkpoint_source", "adapter_name", "preprocessing_version"
    )
    @classmethod
    def _strip_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("foundation adapter identifiers must not be empty")
        return value

    @field_validator("labels", "optional_dependencies")
    @classmethod
    def _validate_non_empty_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if not normalized or any(not value for value in normalized):
            raise ValueError("labels and optional_dependencies must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("labels and optional_dependencies must be unique")
        return normalized

    @field_validator("preprocessing_config")
    @classmethod
    def _validate_preprocessing_config(
        cls, value: dict[str, object]
    ) -> dict[str, object]:
        if not value:
            raise ValueError("preprocessing_config must not be empty")
        return value


class FoundationModelAdapter:
    """Scaffold that supports injected fake outputs but no real backend yet."""

    def __init__(
        self,
        config: FoundationModelAdapterConfig,
        *,
        mock_backend: FoundationMockBackend | None = None,
        allow_real_backend: bool = False,
    ) -> None:
        if mock_backend is None:
            self._refuse_or_validate_real_backend(config, allow_real_backend)
        self.config = config
        self._mock_backend = mock_backend

    @staticmethod
    def _refuse_or_validate_real_backend(
        config: FoundationModelAdapterConfig, allow_real_backend: bool
    ) -> None:
        if not allow_real_backend:
            raise PermissionError(
                "Real foundation-model initialization is disabled by default. "
                "Pass allow_real_backend=True only for an explicit manual/local run."
            )
        missing = [
            package
            for package in config.optional_dependencies
            if find_spec(package) is None
        ]
        if missing:
            raise ImportError(
                "Foundation-model scaffold optional dependencies are unavailable: "
                + ", ".join(missing)
                + ". Install them manually; no packages or weights are downloaded."
            )
        raise NotImplementedError(
            "Real foundation-model inference is not implemented in Phase 9. "
            "Use an injected fake backend for tests; this scaffold never downloads weights."
        )

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def labels(self) -> tuple[str, ...]:
        return self.config.labels

    @property
    def model_version(self) -> str:
        return self.config.checkpoint_source

    @property
    def adapter_name(self) -> str:
        return self.config.adapter_name

    @property
    def preprocessing_version(self) -> str:
        return self.config.preprocessing_version

    @property
    def preprocessing_config(self) -> Mapping[str, object]:
        return self.config.preprocessing_config

    @property
    def uncertainty_strategy(self) -> None:
        return None

    def predict_records(
        self, image_records: Sequence[Mapping[str, object]]
    ) -> PredictionBatch:
        if not image_records:
            raise ValueError("image_records must not be empty")
        if self._mock_backend is None:
            raise NotImplementedError("Real foundation-model inference is not implemented")

        output_rows = self._mock_backend.predict(image_records)
        if len(output_rows) != len(image_records):
            raise ValueError("mock backend output count must match image_records")

        records: list[PredictionRecord] = []
        for row_number, (image_record, output_row) in enumerate(
            zip(image_records, output_rows, strict=True)
        ):
            if len(output_row) != len(self.labels):
                raise ValueError("mock backend output width must match configured labels")
            sample_id = image_record.get("sample_id", image_record.get("image_id"))
            dataset_name = image_record.get("dataset_name")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"image record {row_number} requires a sample_id")
            if not isinstance(dataset_name, str) or not dataset_name.strip():
                raise ValueError(f"image record {row_number} requires a dataset_name")
            probabilities = tuple(float(value) for value in output_row)
            records.append(
                PredictionRecord(
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    model_name=self.model_name,
                    patient_id=_optional_string(image_record, "patient_id", row_number),
                    study_id=_optional_string(image_record, "study_id", row_number),
                    image_path=_optional_string(image_record, "image_path", row_number),
                    label_names=self.labels,
                    probabilities=probabilities,
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
            run_metadata={
                "backend_mode": "injected_mock",
                "real_inference": False,
                "weights_downloaded": False,
            },
        )


def _optional_string(
    image_record: Mapping[str, object], field_name: str, row_number: int
) -> str | None:
    value = image_record.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} in image record {row_number} must be a string")
    return value.strip() if isinstance(value, str) and value.strip() else None
