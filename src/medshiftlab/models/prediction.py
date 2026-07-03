"""Prediction schemas and table adapters for pretrained CXR models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionRecord(BaseModel):
    """Label-wise prediction scores for one chest X-ray record."""

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    scores: dict[str, float | None]
    image_path: str | None = None
    dataset_name: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("image_id", "model_name")
    @classmethod
    def _strip_non_empty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image_id and model_name must not be empty")
        return value

    @field_validator("scores")
    @classmethod
    def _validate_scores(
        cls, scores: dict[str, float | None]
    ) -> dict[str, float | None]:
        for label, score in scores.items():
            if not label.strip():
                raise ValueError("score labels must not be empty")
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError(f"score for {label!r} must be between 0 and 1")
        return scores


class PredictionBatch(BaseModel):
    """Validated predictions from one model for one or more records."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    records: list[PredictionRecord] = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)

    @field_validator("model_name")
    @classmethod
    def _strip_non_empty_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_name must not be empty")
        return value

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if any(not label.strip() for label in labels):
            raise ValueError("labels must contain only non-empty strings")
        return labels

    @model_validator(mode="after")
    def _validate_records(self) -> PredictionBatch:
        for record in self.records:
            if record.model_name != self.model_name:
                raise ValueError("all records must have the batch model_name")
            missing_labels = [
                label for label in self.labels if label not in record.scores
            ]
            if missing_labels:
                missing = ", ".join(missing_labels)
                raise ValueError(
                    f"prediction record {record.image_id!r} is missing labels: {missing}"
                )
        return self


def build_score_mapping_from_predictions(
    predictions: Sequence[PredictionRecord],
    labels: Iterable[str],
) -> dict[str, list[float | None]]:
    """Build label-wise score lists while preserving prediction order."""

    label_list = tuple(labels)
    if not label_list:
        raise ValueError("labels must not be empty")

    scores_by_label: dict[str, list[float | None]] = {
        label: [] for label in label_list
    }
    for prediction in predictions:
        for label in label_list:
            if label not in prediction.scores:
                raise ValueError(
                    f"prediction record {prediction.image_id!r} is missing label {label!r}"
                )
            scores_by_label[label].append(prediction.scores[label])

    return scores_by_label


def build_prediction_table_rows(
    predictions: Sequence[PredictionRecord],
    labels: Iterable[str],
    score_prefix: str = "score_",
) -> list[dict[str, object]]:
    """Convert prediction records into evaluation-compatible score rows."""

    label_list = tuple(labels)
    scores_by_label = build_score_mapping_from_predictions(predictions, label_list)
    rows: list[dict[str, object]] = []

    for row_index, prediction in enumerate(predictions):
        row: dict[str, object] = {
            "image_id": prediction.image_id,
            "image_path": prediction.image_path,
            "dataset_name": prediction.dataset_name,
            "model_name": prediction.model_name,
        }
        for label in label_list:
            row[f"{score_prefix}{label}"] = scores_by_label[label][row_index]
        rows.append(row)

    return rows
