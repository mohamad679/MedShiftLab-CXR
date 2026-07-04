"""CheXpert metadata schema utilities for MedShiftLab-CXR.

This module converts CheXpert-style metadata rows into a validated internal
record representation. It does not load images, train models, or run inference.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medshiftlab.labels.ontology import CXRLabelOntology
from medshiftlab.labels.uncertainty import (
    UncertaintyStrategy,
    transform_chexpert_label_value,
)


CHEXPERT_DATASET_NAME = "CheXpert"
CHEXPERT_PATH_COLUMN = "Path"
CHEXPERT_SEX_COLUMN = "Sex"
CHEXPERT_AGE_COLUMN = "Age"
CHEXPERT_VIEW_COLUMN = "Frontal/Lateral"
CHEXPERT_AP_PA_COLUMN = "AP/PA"


class CheXpertRecord(BaseModel):
    """Validated internal representation of one CheXpert image-level record."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: Literal["CheXpert"] = CHEXPERT_DATASET_NAME
    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    patient_id: str | None = None
    sex: str | None = None
    age: float | None = None
    view_position: str | None = None
    ap_pa: str | None = None
    labels: dict[str, float | None]
    raw_labels: dict[str, Any]

    @field_validator("image_id", "image_path")
    @classmethod
    def _strip_required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required string fields must not be empty")
        return value

    @field_validator("sex", "patient_id", "view_position", "ap_pa")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


def parse_chexpert_record(
    row: Mapping[str, Any],
    ontology: CXRLabelOntology,
    strategy: UncertaintyStrategy | str,
    *,
    soft_value: float = 0.5,
) -> CheXpertRecord:
    """Parse one CheXpert metadata row into a validated record.

    Args:
        row: Mapping from CSV column names to raw values.
        ontology: Validated MedShiftLab-CXR label ontology.
        strategy: CheXpert uncertainty-label handling strategy.
        soft_value: Value used for uncertain labels under U-soft.

    Returns:
        A validated CheXpertRecord.
    """

    image_path = _required_string(row, CHEXPERT_PATH_COLUMN)
    patient_id = infer_chexpert_patient_id(image_path)

    source_to_project = ontology.chexpert_to_project()
    raw_labels: dict[str, Any] = {}
    project_labels: dict[str, float | None] = {}

    for source_label, project_label in source_to_project.items():
        raw_value = row.get(source_label)
        raw_labels[source_label] = raw_value
        project_labels[project_label] = transform_chexpert_label_value(
            raw_value,
            strategy,
            soft_value=soft_value,
        )

    return CheXpertRecord(
        image_id=image_path,
        image_path=image_path,
        patient_id=patient_id,
        sex=_optional_string(row.get(CHEXPERT_SEX_COLUMN)),
        age=_optional_float(row.get(CHEXPERT_AGE_COLUMN)),
        view_position=_optional_string(row.get(CHEXPERT_VIEW_COLUMN)),
        ap_pa=_optional_string(row.get(CHEXPERT_AP_PA_COLUMN)),
        labels=project_labels,
        raw_labels=raw_labels,
    )


def infer_chexpert_patient_id(image_path: str) -> str | None:
    """Infer CheXpert patient ID from a CheXpert-style image path."""

    match = re.search(r"(patient[0-9]+)", image_path)
    if match is None:
        return None
    return match.group(1)


def validate_patient_disjoint_splits(
    split_records: Mapping[str, Sequence[CheXpertRecord]],
) -> None:
    """Validate that present patient IDs occur in only one named split."""

    patient_splits: dict[str, str] = {}

    for split_name, records in split_records.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("split names must not be blank")

        normalized_split_name = split_name.strip()
        for record in records:
            patient_id = record.patient_id
            if patient_id is None:
                continue

            normalized_patient_id = patient_id.strip()
            if not normalized_patient_id:
                raise ValueError("patient IDs must not be blank when present")

            previous_split = patient_splits.get(normalized_patient_id)
            if previous_split is not None and previous_split != normalized_split_name:
                raise ValueError(
                    f"Patient {normalized_patient_id!r} appears in multiple splits: "
                    f"{previous_split!r} and {normalized_split_name!r}"
                )
            patient_splits[normalized_patient_id] = normalized_split_name


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required CheXpert column: {key}")

    string_value = str(value).strip()
    if not string_value:
        raise ValueError(f"Required CheXpert column is blank: {key}")

    return string_value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    string_value = str(value).strip()
    if not string_value or string_value.lower() in {"nan", "none", "null", "na"}:
        return None

    return string_value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null", "na"}:
            return None
        value = stripped

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc

    if math.isnan(parsed):
        return None

    return parsed
