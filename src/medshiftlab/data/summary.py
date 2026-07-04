"""Dataset summary utilities for MedShiftLab-CXR.

This module summarizes validated dataset records. It does not load images,
run model inference, tune thresholds, or fit calibration.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.data.chexpert import CheXpertRecord


class LabelSummary(BaseModel):
    """Summary statistics for one project-level label."""

    model_config = ConfigDict(extra="forbid")

    label_name: str = Field(min_length=1)
    available_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    soft_count: int = Field(ge=0)
    positive_prevalence: float | None = None
    mean_target: float | None = None


class DatasetSummary(BaseModel):
    """Dataset-level summary for validated MedShiftLab-CXR records."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    n_records: int = Field(ge=0)
    n_patients: int | None = Field(default=None, ge=0)
    n_records_without_patient_id: int = Field(ge=0)
    labels: dict[str, LabelSummary]


def summarize_chexpert_records(records: Sequence[CheXpertRecord]) -> DatasetSummary:
    """Create a dataset summary from CheXpert records."""

    if not records:
        raise ValueError("Cannot summarize an empty record collection")

    dataset_names = {record.dataset_name for record in records}
    if len(dataset_names) != 1:
        raise ValueError(f"Expected one dataset name, received: {sorted(dataset_names)}")

    dataset_name = dataset_names.pop()
    patient_ids = [record.patient_id for record in records if record.patient_id is not None]
    n_records_without_patient_id = sum(1 for record in records if record.patient_id is None)

    label_names = _collect_label_names(records)
    label_summaries = {
        label_name: _summarize_label(records, label_name)
        for label_name in label_names
    }

    return DatasetSummary(
        dataset_name=dataset_name,
        n_records=len(records),
        n_patients=len(set(patient_ids)) if patient_ids else None,
        n_records_without_patient_id=n_records_without_patient_id,
        labels=label_summaries,
    )


def _collect_label_names(records: Sequence[CheXpertRecord]) -> tuple[str, ...]:
    label_names: list[str] = []

    for record in records:
        for label_name in record.labels:
            if label_name not in label_names:
                label_names.append(label_name)

    return tuple(label_names)


def _summarize_label(records: Sequence[CheXpertRecord], label_name: str) -> LabelSummary:
    values = [record.labels.get(label_name) for record in records]

    missing_count = sum(value is None for value in values)
    available_values = [float(value) for value in values if value is not None]

    positive_count = sum(value == 1.0 for value in available_values)
    negative_count = sum(value == 0.0 for value in available_values)
    soft_count = sum(value not in {0.0, 1.0} for value in available_values)

    available_count = len(available_values)
    positive_prevalence = positive_count / available_count if available_count else None
    mean_target = sum(available_values) / available_count if available_count else None

    return LabelSummary(
        label_name=label_name,
        available_count=available_count,
        missing_count=missing_count,
        positive_count=positive_count,
        negative_count=negative_count,
        soft_count=soft_count,
        positive_prevalence=positive_prevalence,
        mean_target=mean_target,
    )
