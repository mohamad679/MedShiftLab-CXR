"""Prediction schemas and table adapters for pretrained CXR models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


PREDICTION_SCHEMA_VERSION = "medshiftlab.prediction.v1"


class PredictionRecord(BaseModel):
    """Versioned prediction payload for one chest X-ray sample."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sample_id", "image_id"),
    )
    model_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    label_names: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("label_names", "labels"),
    )
    probabilities: tuple[float | None, ...] = Field(min_length=1)
    logits: tuple[float | None, ...] | None = None
    thresholds: float | tuple[float, ...] | None = None
    thresholded_predictions: tuple[int | None, ...] | None = None
    patient_id: str | None = None
    study_id: str | None = None
    image_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_scores(
        cls, data: object
    ) -> object:
        if not isinstance(data, Mapping):
            return data

        payload = dict(data)
        if "scores" not in payload:
            return payload
        if "probabilities" in payload or "label_names" in payload or "labels" in payload:
            raise ValueError(
                "scores cannot be combined with explicit label_names/probabilities"
            )

        scores = payload.pop("scores")
        if not isinstance(scores, Mapping) or not scores:
            raise ValueError("scores must be a non-empty mapping")

        payload["label_names"] = tuple(str(label) for label in scores)
        payload["probabilities"] = tuple(scores.values())
        return payload

    @field_validator("sample_id", "model_name", "dataset_name")
    @classmethod
    def _strip_non_empty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sample_id, model_name, and dataset_name must not be empty")
        return value

    @field_validator("patient_id", "study_id", "image_path")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("label_names")
    @classmethod
    def _validate_label_names(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for label in labels:
            stripped = label.strip()
            if not stripped:
                raise ValueError("label_names must contain only non-empty strings")
            normalized.append(stripped)
        if len(set(normalized)) != len(normalized):
            raise ValueError("label_names must be unique")
        return tuple(normalized)

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(
        cls, values: tuple[float | None, ...]
    ) -> tuple[float | None, ...]:
        for value in values:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must be between 0 and 1")
        return values

    @field_validator("thresholds", mode="before")
    @classmethod
    def _coerce_thresholds(
        cls, value: object
    ) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(
        cls, value: float | tuple[float, ...] | None
    ) -> float | tuple[float, ...] | None:
        if value is None:
            return None
        if isinstance(value, tuple):
            for threshold in value:
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("thresholds must be between 0 and 1")
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        return value

    @field_validator("thresholded_predictions")
    @classmethod
    def _validate_thresholded_predictions(
        cls, values: tuple[int | None, ...] | None
    ) -> tuple[int | None, ...] | None:
        if values is None:
            return None
        for value in values:
            if value is not None and value not in (0, 1):
                raise ValueError("thresholded_predictions must contain only 0, 1, or None")
        return values

    @model_validator(mode="after")
    def _validate_vector_lengths(self) -> PredictionRecord:
        label_count = len(self.label_names)
        if len(self.probabilities) != label_count:
            raise ValueError("probabilities must align with label_names")
        if self.logits is not None and len(self.logits) != label_count:
            raise ValueError("logits must align with label_names")
        if isinstance(self.thresholds, tuple) and len(self.thresholds) != label_count:
            raise ValueError("thresholds must align with label_names")
        if self.thresholded_predictions is not None:
            if self.thresholds is None:
                raise ValueError(
                    "thresholded_predictions must not exist unless thresholds are provided"
                )
            if len(self.thresholded_predictions) != label_count:
                raise ValueError(
                    "thresholded_predictions must align with label_names"
                )
        return self

    @property
    def image_id(self) -> str:
        """Compatibility alias for pre-Phase 4 code paths."""

        return self.sample_id

    @property
    def scores(self) -> dict[str, float | None]:
        """Compatibility mapping for evaluation utilities."""

        return dict(zip(self.label_names, self.probabilities, strict=True))

    @property
    def threshold_mapping(self) -> dict[str, float] | None:
        """Return per-label thresholds regardless of scalar or vector storage."""

        if self.thresholds is None:
            return None
        if isinstance(self.thresholds, tuple):
            return dict(zip(self.label_names, self.thresholds, strict=True))
        return {label: self.thresholds for label in self.label_names}


class PredictionBatch(BaseModel):
    """Validated predictions from one model for one or more records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(default=PREDICTION_SCHEMA_VERSION, min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    preprocessing_config: dict[str, object] = Field(default_factory=dict)
    records: list[PredictionRecord] = Field(min_length=1)
    label_names: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("label_names", "labels"),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    uncertainty_strategy: str | None = None
    run_metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "schema_version",
        "model_name",
        "model_version",
        "adapter_name",
        "preprocessing_version",
    )
    @classmethod
    def _strip_non_empty_batch_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(
                "schema_version, model_name, model_version, adapter_name, and "
                "preprocessing_version must not be empty"
            )
        return value

    @field_validator("uncertainty_strategy")
    @classmethod
    def _strip_optional_uncertainty_strategy(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("label_names")
    @classmethod
    def _validate_label_names(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for label in labels:
            stripped = label.strip()
            if not stripped:
                raise ValueError("label_names must contain only non-empty strings")
            normalized.append(stripped)
        if len(set(normalized)) != len(normalized):
            raise ValueError("label_names must be unique")
        return tuple(normalized)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_records(self) -> PredictionBatch:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PREDICTION_SCHEMA_VERSION!r}"
            )
        if not self.preprocessing_config:
            raise ValueError("preprocessing_config must not be empty")

        seen_sample_ids: set[str] = set()
        for record in self.records:
            if record.model_name != self.model_name:
                raise ValueError("all records must have the batch model_name")
            if record.label_names != self.label_names:
                raise ValueError("all records must share the batch label_names")
            if record.sample_id in seen_sample_ids:
                raise ValueError(
                    f"duplicate prediction sample_id: {record.sample_id!r}"
                )
            seen_sample_ids.add(record.sample_id)
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        """Compatibility alias for pre-Phase 4 code paths."""

        return self.label_names


def build_score_mapping_from_predictions(
    predictions: Sequence[PredictionRecord],
    labels: Iterable[str],
) -> dict[str, list[float | None]]:
    """Build label-wise score lists while preserving prediction order."""

    label_list = tuple(label.strip() for label in labels)
    if not label_list or any(not label for label in label_list):
        raise ValueError("labels must not be empty")

    scores_by_label: dict[str, list[float | None]] = {
        label: [] for label in label_list
    }
    for prediction in predictions:
        for label in label_list:
            if label not in prediction.scores:
                raise ValueError(
                    f"prediction record {prediction.sample_id!r} is missing label {label!r}"
                )
            scores_by_label[label].append(prediction.scores[label])

    return scores_by_label


def build_prediction_table_rows(
    predictions: Sequence[PredictionRecord],
    labels: Iterable[str],
    score_prefix: str = "score_",
) -> list[dict[str, object]]:
    """Convert prediction records into evaluation-compatible score rows."""

    label_list = tuple(label.strip() for label in labels)
    scores_by_label = build_score_mapping_from_predictions(predictions, label_list)
    rows: list[dict[str, object]] = []

    for row_index, prediction in enumerate(predictions):
        row: dict[str, object] = {
            "sample_id": prediction.sample_id,
            "image_id": prediction.sample_id,
            "image_path": prediction.image_path,
            "dataset_name": prediction.dataset_name,
            "model_name": prediction.model_name,
        }
        for label in label_list:
            row[f"{score_prefix}{label}"] = scores_by_label[label][row_index]
        rows.append(row)

    return rows
