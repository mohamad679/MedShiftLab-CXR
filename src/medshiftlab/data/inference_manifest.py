"""CSV manifest utilities for bounded local baseline inference."""

from __future__ import annotations

import csv
from pathlib import Path, PureWindowsPath

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class InferenceManifestRecord(BaseModel):
    """One non-sensitive inference input record with relative image location."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sample_id", "image_id"),
    )
    dataset_name: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    patient_id: str | None = None
    study_id: str | None = None

    @field_validator("sample_id", "dataset_name", "image_path")
    @classmethod
    def _strip_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manifest string fields must not be empty")
        return value

    @field_validator("patient_id", "study_id")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("image_path")
    @classmethod
    def _validate_relative_image_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("manifest image_path must be relative")
        if ".." in path.parts:
            raise ValueError("manifest image_path must remain inside the dataset directory")
        return value


def load_inference_manifest_csv(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    limit: int | None = None,
) -> list[InferenceManifestRecord]:
    """Load and validate a bounded inference manifest CSV."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Inference manifest CSV not found: {manifest_path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Inference manifest CSV must include a header row")

        fieldnames = set(reader.fieldnames)
        if not {"sample_id", "image_id"} & fieldnames:
            raise ValueError(
                "Inference manifest CSV must contain either 'sample_id' or 'image_id'"
            )
        if "image_path" not in fieldnames:
            raise ValueError("Inference manifest CSV must contain an 'image_path' column")

        records: list[InferenceManifestRecord] = []
        for row_index, row in enumerate(reader):
            payload = {key: value for key, value in row.items() if key is not None}
            if dataset_name is not None and not payload.get("dataset_name", "").strip():
                payload["dataset_name"] = dataset_name
            try:
                record = InferenceManifestRecord.model_validate(payload)
            except Exception as error:
                raise ValueError(
                    f"Inference manifest row {row_index + 2} is invalid: {error}"
                ) from error
            if dataset_name is not None and record.dataset_name != dataset_name:
                raise ValueError(
                    f"Inference manifest row {row_index + 2} uses dataset "
                    f"{record.dataset_name!r}, expected {dataset_name!r}"
                )
            records.append(record)
            if limit is not None and len(records) == limit:
                break

    if not records:
        raise ValueError("Inference manifest CSV must contain at least one record row")
    return records
