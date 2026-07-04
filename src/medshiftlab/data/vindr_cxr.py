"""VinDr-CXR metadata schema utilities for MedShiftLab-CXR.

This module converts image-level VinDr-CXR-style metadata rows into validated
internal records. It does not load images, aggregate bounding boxes, train
models, or run inference.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medshiftlab.labels.ontology import CXRLabelOntology


VINDR_CXR_DATASET_NAME = "VinDr-CXR"
VINDR_IMAGE_ID_COLUMNS = ("image_id", "ImageID", "imageId")
VINDR_IMAGE_PATH_COLUMNS = ("image_path", "Path", "path")
VINDR_SPLIT_COLUMNS = ("split", "Split")
VINDR_VIEW_POSITION_COLUMNS = ("view_position", "ViewPosition", "view")


class VinDrCXRRecord(BaseModel):
    """Validated internal representation of one VinDr-CXR image-level record."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: Literal["VinDr-CXR"] = VINDR_CXR_DATASET_NAME
    image_id: str = Field(min_length=1)
    image_path: str | None = None
    patient_id: str | None = None
    split: str | None = None
    view_position: str | None = None
    labels: dict[str, float | None]
    raw_labels: dict[str, Any]

    @field_validator("image_id")
    @classmethod
    def _strip_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image_id must not be empty")
        return value

    @field_validator("image_path", "patient_id", "split", "view_position")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


def parse_vindr_cxr_record(
    row: Mapping[str, Any],
    ontology: CXRLabelOntology,
) -> VinDrCXRRecord:
    """Parse one image-level VinDr-CXR-style metadata row.

    The first implementation expects image-level binary labels. Raw VinDr-CXR
    annotation aggregation can be added later as a separate, explicit step.
    """

    image_id = _first_required_string(row, VINDR_IMAGE_ID_COLUMNS)
    image_path = _first_optional_string(row, VINDR_IMAGE_PATH_COLUMNS)
    split = _first_optional_string(row, VINDR_SPLIT_COLUMNS)
    view_position = _first_optional_string(row, VINDR_VIEW_POSITION_COLUMNS)

    source_to_project = ontology.vindr_to_project()
    raw_labels: dict[str, Any] = {}
    project_labels: dict[str, float | None] = {}

    for source_label, project_label in source_to_project.items():
        raw_value = row.get(source_label)
        raw_labels[source_label] = raw_value
        project_labels[project_label] = _normalize_binary_label_value(raw_value)

    return VinDrCXRRecord(
        image_id=image_id,
        image_path=image_path,
        split=split,
        view_position=view_position,
        labels=project_labels,
        raw_labels=raw_labels,
    )


def _first_required_string(row: Mapping[str, Any], candidate_keys: tuple[str, ...]) -> str:
    for key in candidate_keys:
        value = _optional_string(row.get(key))
        if value is not None:
            return value

    joined_keys = ", ".join(candidate_keys)
    raise ValueError(f"Missing required VinDr-CXR image ID column. Tried: {joined_keys}")


def _first_optional_string(row: Mapping[str, Any], candidate_keys: tuple[str, ...]) -> str | None:
    for key in candidate_keys:
        value = _optional_string(row.get(key))
        if value is not None:
            return value
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    string_value = str(value).strip()
    if not string_value or string_value.lower() in {"nan", "none", "null", "na"}:
        return None

    return string_value


def _normalize_binary_label_value(value: Any) -> float | None:
    """Normalize VinDr-CXR image-level labels into 1.0, 0.0, or None."""

    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null", "na"}:
            return None
        value = stripped

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid VinDr-CXR binary label value: {value!r}") from exc

    if numeric_value == 1.0:
        return 1.0

    if numeric_value == 0.0:
        return 0.0

    raise ValueError(
        "VinDr-CXR image-level labels must be binary 1/0 or missing. "
        f"Received: {value!r}"
    )
