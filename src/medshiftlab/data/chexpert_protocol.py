"""CheXpert internal-protocol scaffolding utilities.

This module prepares patient-disjoint split manifests and uncertainty-specific
label tables from CheXpert-style metadata. It does not load raw images, run
model inference, or write outputs unless explicitly requested by a caller.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medshiftlab.data.chexpert import CheXpertRecord, validate_patient_disjoint_splits
from medshiftlab.data.chexpert_loader import load_chexpert_metadata_csv
from medshiftlab.labels import (
    CXRLabelOntology,
    UncertaintyStrategy,
    load_default_label_ontology,
    parse_uncertainty_strategy,
)


CHEXPERT_PROTOCOL_DATASET_NAME = "chexpert"
DEFAULT_CHEXPERT_SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_CHEXPERT_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_CHEXPERT_PROTOCOL_SEED = 2026
DEFAULT_CHEXPERT_PROTOCOL_LIMIT = 256
MAX_SAFE_CHEXPERT_PROTOCOL_LIMIT_WITHOUT_OVERRIDE = 4096


class CheXpertSplitConfig(BaseModel):
    """Validated patient-level split policy for CheXpert internal protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_names: tuple[str, ...] = DEFAULT_CHEXPERT_SPLIT_NAMES
    split_fractions: tuple[float, ...] = DEFAULT_CHEXPERT_SPLIT_FRACTIONS
    seed: int = DEFAULT_CHEXPERT_PROTOCOL_SEED

    @field_validator("split_names")
    @classmethod
    def _validate_split_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2:
            raise ValueError("split_names must contain at least two entries")

        normalized: list[str] = []
        for split_name in value:
            stripped = split_name.strip()
            if not stripped:
                raise ValueError("split_names must not contain blank values")
            normalized.append(stripped)

        if len(set(normalized)) != len(normalized):
            raise ValueError("split_names must be unique")
        return tuple(normalized)

    @field_validator("split_fractions")
    @classmethod
    def _validate_split_fractions(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("split_fractions must contain at least two entries")
        if any(fraction <= 0.0 for fraction in value):
            raise ValueError("split_fractions must contain only positive values")
        return tuple(float(fraction) for fraction in value)

    @model_validator(mode="after")
    def _validate_lengths_and_sum(self) -> CheXpertSplitConfig:
        if len(self.split_names) != len(self.split_fractions):
            raise ValueError("split_names and split_fractions must have the same length")
        if not math.isclose(sum(self.split_fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("split_fractions must sum to 1.0")
        return self


class CheXpertSplitManifestRow(BaseModel):
    """One patient-disjoint split assignment for a CheXpert sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    dataset_name: Literal["chexpert"] = CHEXPERT_PROTOCOL_DATASET_NAME
    patient_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    sex: str | None = None
    age: float | None = None
    view_position: str | None = None
    ap_pa: str | None = None

    @field_validator("sample_id", "patient_id", "split", "image_path")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required split-manifest string fields must not be empty")
        return value

    @field_validator("sex", "view_position", "ap_pa")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("image_path")
    @classmethod
    def _validate_relative_image_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("split-manifest image_path must be relative")
        if ".." in path.parts:
            raise ValueError("split-manifest image_path must remain inside the dataset")
        return value


class CheXpertLabelTableRow(BaseModel):
    """One Phase 6-compatible label row with protocol provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    dataset_name: Literal["chexpert"] = CHEXPERT_PROTOCOL_DATASET_NAME
    patient_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    uncertainty_strategy: str = Field(min_length=1)
    sex: str | None = None
    age: float | None = None
    view_position: str | None = None
    ap_pa: str | None = None
    labels: dict[str, float | None] = Field(min_length=1)

    @field_validator("sample_id", "patient_id", "split", "uncertainty_strategy")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required label-table string fields must not be empty")
        return value

    @field_validator("sex", "view_position", "ap_pa")
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("labels")
    @classmethod
    def _validate_labels(
        cls,
        value: dict[str, float | None],
    ) -> dict[str, float | None]:
        if not value:
            raise ValueError("labels must not be empty")

        normalized: dict[str, float | None] = {}
        for label_name, label_value in value.items():
            stripped = label_name.strip()
            if not stripped:
                raise ValueError("labels must use non-empty label names")
            normalized[stripped] = None if label_value is None else float(label_value)
        return normalized

    def to_flat_row(self, *, label_names: Sequence[str]) -> dict[str, object]:
        """Convert the row into a CSV/evaluation-compatible flat mapping."""

        row: dict[str, object] = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "patient_id": self.patient_id,
            "split": self.split,
            "uncertainty_strategy": self.uncertainty_strategy,
            "sex": self.sex,
            "age": self.age,
            "view_position": self.view_position,
            "ap_pa": self.ap_pa,
        }
        for label_name in label_names:
            if label_name not in self.labels:
                raise ValueError(f"Missing label {label_name!r} in label-table row")
            row[label_name] = self.labels[label_name]
        return row


class CheXpertInternalProtocolPreparation(BaseModel):
    """Prepared split manifest and label tables for a bounded protocol cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_config: CheXpertSplitConfig
    label_names: tuple[str, ...]
    split_manifest_rows: tuple[CheXpertSplitManifestRow, ...]
    label_table_rows_by_strategy: dict[str, tuple[CheXpertLabelTableRow, ...]]


def prepare_chexpert_internal_protocol(
    metadata_csv_path: str | Path,
    *,
    split_config: CheXpertSplitConfig | None = None,
    uncertainty_strategies: Sequence[UncertaintyStrategy | str] = tuple(
        strategy for strategy in UncertaintyStrategy
    ),
    ontology: CXRLabelOntology | None = None,
    max_rows: int | None = None,
    soft_value: float = 0.5,
) -> CheXpertInternalProtocolPreparation:
    """Prepare a bounded split manifest and label tables from local metadata."""

    parsed_split_config = split_config or CheXpertSplitConfig()
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    parsed_strategies = tuple(
        parse_uncertainty_strategy(strategy) for strategy in uncertainty_strategies
    )
    if not parsed_strategies:
        raise ValueError("uncertainty_strategies must contain at least one strategy")

    label_ontology = ontology or load_default_label_ontology()
    base_records = load_chexpert_metadata_csv(
        metadata_csv_path,
        label_ontology,
        UncertaintyStrategy.IGNORE,
        soft_value=soft_value,
        max_rows=max_rows,
    )
    split_records = assign_patient_level_splits(
        base_records,
        config=parsed_split_config,
    )
    split_manifest_rows = tuple(build_split_manifest_rows(split_records))
    split_by_sample_id = {
        row.sample_id: row.split for row in split_manifest_rows
    }
    label_names = tuple(base_records[0].labels) if base_records else ()

    label_table_rows_by_strategy: dict[str, tuple[CheXpertLabelTableRow, ...]] = {}
    for strategy in parsed_strategies:
        records = (
            base_records
            if strategy is UncertaintyStrategy.IGNORE
            else load_chexpert_metadata_csv(
                metadata_csv_path,
                label_ontology,
                strategy,
                soft_value=soft_value,
                max_rows=max_rows,
            )
        )
        label_table_rows_by_strategy[strategy.value] = tuple(
            materialize_chexpert_label_table_rows(
                records,
                uncertainty_strategy=strategy,
                split_by_sample_id=split_by_sample_id,
                label_names=label_names,
            )
        )

    return CheXpertInternalProtocolPreparation(
        split_config=parsed_split_config,
        label_names=label_names,
        split_manifest_rows=split_manifest_rows,
        label_table_rows_by_strategy=label_table_rows_by_strategy,
    )


def assign_patient_level_splits(
    records: Sequence[CheXpertRecord],
    *,
    config: CheXpertSplitConfig | None = None,
) -> dict[str, list[CheXpertRecord]]:
    """Assign every patient to exactly one split under a deterministic seed."""

    if not records:
        raise ValueError("records must not be empty")

    split_config = config or CheXpertSplitConfig()
    records_by_patient: dict[str, list[CheXpertRecord]] = defaultdict(list)
    for record in records:
        patient_id = _require_patient_id(record)
        records_by_patient[patient_id].append(record)

    patient_ids = sorted(records_by_patient)
    shuffled_patient_ids = patient_ids[:]
    random.Random(split_config.seed).shuffle(shuffled_patient_ids)

    counts = _calculate_split_counts(
        n_patients=len(shuffled_patient_ids),
        fractions=split_config.split_fractions,
    )

    split_records: dict[str, list[CheXpertRecord]] = {
        split_name: [] for split_name in split_config.split_names
    }
    offset = 0
    for split_name, count in zip(split_config.split_names, counts, strict=True):
        for patient_id in shuffled_patient_ids[offset : offset + count]:
            split_records[split_name].extend(records_by_patient[patient_id])
        offset += count

    validate_patient_disjoint_splits(split_records)
    return split_records


def build_split_manifest_rows(
    split_records: Mapping[str, Sequence[CheXpertRecord]],
) -> list[CheXpertSplitManifestRow]:
    """Flatten split assignments into stable manifest rows."""

    validate_patient_disjoint_splits(split_records)

    rows: list[CheXpertSplitManifestRow] = []
    for split_name, records in split_records.items():
        normalized_split_name = split_name.strip()
        for record in records:
            rows.append(
                CheXpertSplitManifestRow(
                    sample_id=record.image_id,
                    patient_id=_require_patient_id(record),
                    split=normalized_split_name,
                    image_path=record.image_path,
                    sex=record.sex,
                    age=record.age,
                    view_position=record.view_position,
                    ap_pa=record.ap_pa,
                )
            )
    return rows


def materialize_chexpert_label_table_rows(
    records: Sequence[CheXpertRecord],
    *,
    uncertainty_strategy: UncertaintyStrategy | str,
    split_by_sample_id: Mapping[str, str],
    label_names: Sequence[str] | None = None,
) -> list[CheXpertLabelTableRow]:
    """Materialize Phase 6-compatible label-table rows for one strategy."""

    if not records:
        raise ValueError("records must not be empty")

    parsed_strategy = parse_uncertainty_strategy(uncertainty_strategy)
    selected_label_names = tuple(label_names or records[0].labels)
    rows: list[CheXpertLabelTableRow] = []

    for record in records:
        split_name = split_by_sample_id.get(record.image_id)
        if split_name is None:
            raise ValueError(
                f"Missing split assignment for sample_id {record.image_id!r}"
            )

        missing_labels = [
            label_name
            for label_name in selected_label_names
            if label_name not in record.labels
        ]
        if missing_labels:
            raise ValueError(
                "CheXpert record is missing required label(s): "
                + ", ".join(missing_labels)
            )

        rows.append(
            CheXpertLabelTableRow(
                sample_id=record.image_id,
                patient_id=_require_patient_id(record),
                split=split_name,
                uncertainty_strategy=parsed_strategy.value,
                sex=record.sex,
                age=record.age,
                view_position=record.view_position,
                ap_pa=record.ap_pa,
                labels={
                    label_name: record.labels[label_name]
                    for label_name in selected_label_names
                },
            )
        )

    return rows


def write_chexpert_split_manifest_csv(
    rows: Sequence[CheXpertSplitManifestRow],
    output_path: str | Path,
) -> Path:
    """Write a split manifest CSV with relative sample/image identifiers only."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "sample_id",
                "dataset_name",
                "patient_id",
                "split",
                "image_path",
                "sex",
                "age",
                "view_position",
                "ap_pa",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))
    return path


def write_chexpert_label_table_csv(
    rows: Sequence[CheXpertLabelTableRow],
    output_path: str | Path,
    *,
    label_names: Sequence[str],
) -> Path:
    """Write a Phase 6-compatible label-table CSV for one uncertainty strategy."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "sample_id",
                "dataset_name",
                "patient_id",
                "split",
                "uncertainty_strategy",
                "sex",
                "age",
                "view_position",
                "ap_pa",
                *label_names,
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_flat_row(label_names=label_names))
    return path


def _calculate_split_counts(
    *,
    n_patients: int,
    fractions: Sequence[float],
) -> tuple[int, ...]:
    if n_patients <= 0:
        raise ValueError("n_patients must be positive")

    raw_counts = [n_patients * fraction for fraction in fractions]
    base_counts = [int(math.floor(raw_count)) for raw_count in raw_counts]
    remainder = n_patients - sum(base_counts)

    ranked_indices = sorted(
        range(len(fractions)),
        key=lambda index: (raw_counts[index] - base_counts[index], -index),
        reverse=True,
    )
    for index in ranked_indices[:remainder]:
        base_counts[index] += 1

    return tuple(base_counts)


def _require_patient_id(record: CheXpertRecord) -> str:
    patient_id = record.patient_id
    if patient_id is None:
        raise ValueError(
            f"CheXpert internal protocol requires patient_id for sample_id "
            f"{record.image_id!r}"
        )

    normalized_patient_id = patient_id.strip()
    if not normalized_patient_id:
        raise ValueError("patient_id must not be blank for CheXpert internal protocol")
    return normalized_patient_id
