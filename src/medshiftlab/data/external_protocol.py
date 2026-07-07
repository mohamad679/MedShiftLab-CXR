"""External-validation setup scaffolding for MedShiftLab-CXR.

This module prepares bounded external-cohort manifests and harmonized label
tables from local metadata. It does not run model inference, evaluate models,
or write outputs unless explicitly requested by a caller.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medshiftlab.data.registry import get_dataset_registry_entry
from medshiftlab.labels import CXRLabelOntology, load_default_label_ontology


DEFAULT_EXTERNAL_PROTOCOL_LIMIT = 256
MAX_SAFE_EXTERNAL_PROTOCOL_LIMIT_WITHOUT_OVERRIDE = 4096
SUPPORTED_EXTERNAL_DATASET_NAMES = ("mimic_cxr_jpg", "vindr_cxr")


class ExternalValidationProtocolConfig(BaseModel):
    """Validated external-validation protocol freeze for one dataset candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    dataset: str = Field(min_length=1)
    role: Literal["external_validation_candidate"]
    external_validation_only: Literal[True]
    allow_threshold_tuning: Literal[False] = False
    allow_hyperparameter_tuning: Literal[False] = False
    allow_model_selection: Literal[False] = False
    allow_protocol_edit_after_results: Literal[False] = False
    label_set_reference: str = Field(min_length=1)
    label_harmonization_reference: str = Field(min_length=1)
    prediction_schema_version: str = Field(min_length=1)
    evaluation_metrics_reference: str = Field(min_length=1)
    safe_default_limit: int = Field(
        default=DEFAULT_EXTERNAL_PROTOCOL_LIMIT,
        gt=0,
        le=MAX_SAFE_EXTERNAL_PROTOCOL_LIMIT_WITHOUT_OVERRIDE,
    )
    local_private_output_dir: str = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @field_validator(
        "dataset",
        "label_set_reference",
        "label_harmonization_reference",
        "prediction_schema_version",
        "evaluation_metrics_reference",
        "local_private_output_dir",
    )
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("protocol string fields must not be empty")
        return value

    @field_validator("notes")
    @classmethod
    def _normalize_notes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                raise ValueError("notes must not contain blank entries")
            normalized.append(stripped)
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_dataset_role(self) -> ExternalValidationProtocolConfig:
        entry = get_dataset_registry_entry(self.dataset)
        if not entry.external_validation_only:
            raise ValueError(
                f"Dataset {entry.name!r} is not marked external_validation_only"
            )
        return self


class ExternalLabelHarmonizationConfig(BaseModel):
    """Dataset-specific mapping from source label columns into project labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    dataset: str = Field(min_length=1)
    label_set_reference: str = Field(min_length=1)
    mapped_labels: dict[str, str] = Field(min_length=1)
    sample_id_columns: tuple[str, ...] = ("sample_id", "image_id")
    image_path_columns: tuple[str, ...] = ("image_path", "path", "Path")
    patient_id_columns: tuple[str, ...] = ("patient_id",)
    study_id_columns: tuple[str, ...] = ("study_id",)
    split_columns: tuple[str, ...] = ("split",)
    view_position_columns: tuple[str, ...] = ("view_position",)
    known_metadata_columns: tuple[str, ...] = ()

    @field_validator(
        "dataset",
        "label_set_reference",
    )
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("harmonization string fields must not be empty")
        return value

    @field_validator(
        "sample_id_columns",
        "image_path_columns",
        "patient_id_columns",
        "study_id_columns",
        "split_columns",
        "view_position_columns",
        "known_metadata_columns",
    )
    @classmethod
    def _normalize_string_tuples(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                raise ValueError("column-name tuples must not contain blank values")
            normalized.append(stripped)
        if len(set(normalized)) != len(normalized):
            raise ValueError("column-name tuples must not contain duplicates")
        return tuple(normalized)

    @field_validator("mapped_labels")
    @classmethod
    def _validate_mapped_labels(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("mapped_labels must not be empty")

        normalized: dict[str, str] = {}
        project_labels: list[str] = []
        for source_label, project_label in value.items():
            normalized_source = source_label.strip()
            normalized_project = project_label.strip()
            if not normalized_source or not normalized_project:
                raise ValueError("mapped_labels must use non-empty source/project labels")
            normalized[normalized_source] = normalized_project
            project_labels.append(normalized_project)

        if len(set(normalized.values())) != len(project_labels):
            raise ValueError("mapped_labels must not map multiple source labels to one project label")
        return normalized


class ExternalValidationManifestRow(BaseModel):
    """Validated manifest row for an external validation candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    patient_id: str | None = None
    study_id: str | None = None
    split: str | None = None
    view_position: str | None = None

    @field_validator("sample_id", "dataset_name", "image_path")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manifest string fields must not be empty")
        return value

    @field_validator("patient_id", "study_id", "split", "view_position")
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
            raise ValueError("manifest image_path must be relative")
        if ".." in path.parts:
            raise ValueError("manifest image_path must remain inside the dataset directory")
        return value


class ExternalValidationLabelTableRow(BaseModel):
    """One Phase 6-compatible external label row with harmonization provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    patient_id: str | None = None
    study_id: str | None = None
    split: str | None = None
    view_position: str | None = None
    harmonization_reference: str = Field(min_length=1)
    labels: dict[str, float | None] = Field(min_length=1)

    @field_validator("sample_id", "dataset_name", "harmonization_reference")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label-table string fields must not be empty")
        return value

    @field_validator("patient_id", "study_id", "split", "view_position")
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
        """Convert the row into a Phase 6-compatible flat CSV mapping."""

        row: dict[str, object] = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "split": self.split,
            "view_position": self.view_position,
            "harmonization_reference": self.harmonization_reference,
        }
        for label_name in label_names:
            if label_name not in self.labels:
                raise ValueError(f"Missing label {label_name!r} in label-table row")
            row[label_name] = self.labels[label_name]
        return row


class ExternalValidationPreparation(BaseModel):
    """Prepared manifest and label table for one external dataset candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_config: ExternalValidationProtocolConfig
    harmonization_config: ExternalLabelHarmonizationConfig
    label_names: tuple[str, ...]
    manifest_rows: tuple[ExternalValidationManifestRow, ...]
    label_table_rows: tuple[ExternalValidationLabelTableRow, ...]
    excluded_source_labels: tuple[str, ...] = ()
    overlapping_internal_patient_ids: tuple[str, ...] = ()


def default_external_protocol_config_path(dataset_name: str) -> Path:
    """Return the repository-default external protocol config path."""

    entry = get_dataset_registry_entry(dataset_name)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "protocol" / f"{entry.name}_external.yaml"


def default_external_label_harmonization_config_path(dataset_name: str) -> Path:
    """Return the repository-default external harmonization config path."""

    entry = get_dataset_registry_entry(dataset_name)
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "configs"
        / "protocol"
        / "harmonization"
        / f"{entry.name}_labels.yaml"
    )


def load_external_validation_protocol_config(
    path: str | Path,
) -> ExternalValidationProtocolConfig:
    """Load a dataset-specific external-validation protocol config YAML."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"External validation protocol config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw_data: Any = yaml.safe_load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("External validation protocol config must be a mapping")
    return ExternalValidationProtocolConfig.model_validate(raw_data)


def load_external_label_harmonization_config(
    path: str | Path,
) -> ExternalLabelHarmonizationConfig:
    """Load a dataset-specific external label harmonization config YAML."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"External label harmonization config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw_data: Any = yaml.safe_load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("External label harmonization config must be a mapping")
    return ExternalLabelHarmonizationConfig.model_validate(raw_data)


def prepare_external_validation_protocol(
    dataset_name: str,
    metadata_csv_path: str | Path,
    *,
    protocol_config_path: str | Path | None = None,
    label_mapping_config_path: str | Path | None = None,
    ontology: CXRLabelOntology | None = None,
    max_rows: int | None = None,
    internal_manifest_path: str | Path | None = None,
) -> ExternalValidationPreparation:
    """Prepare bounded external manifest and harmonized label-table rows."""

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    protocol_config = load_external_validation_protocol_config(
        protocol_config_path
        or default_external_protocol_config_path(dataset_name)
    )
    harmonization_config = load_external_label_harmonization_config(
        label_mapping_config_path
        or default_external_label_harmonization_config_path(dataset_name)
    )

    normalized_dataset_name = get_dataset_registry_entry(dataset_name).name
    if protocol_config.dataset != normalized_dataset_name:
        raise ValueError(
            "Protocol config dataset does not match requested dataset: "
            f"{protocol_config.dataset!r} != {normalized_dataset_name!r}"
        )
    if harmonization_config.dataset != normalized_dataset_name:
        raise ValueError(
            "Label harmonization config dataset does not match requested dataset: "
            f"{harmonization_config.dataset!r} != {normalized_dataset_name!r}"
        )

    label_ontology = ontology or load_default_label_ontology()
    expected_label_names = label_ontology.all_project_labels
    mapped_project_labels = tuple(harmonization_config.mapped_labels.values())
    missing_project_labels = [
        label_name
        for label_name in expected_label_names
        if label_name not in mapped_project_labels
    ]
    if missing_project_labels:
        raise ValueError(
            "External label harmonization config is missing project label mapping(s): "
            + ", ".join(missing_project_labels)
        )

    metadata_path = Path(metadata_csv_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"External metadata CSV not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("External metadata CSV must include a header row")

        fieldnames = tuple(reader.fieldnames)
        _validate_required_harmonization_columns(
            fieldnames=fieldnames,
            config=harmonization_config,
        )
        excluded_source_labels = tuple(
            sorted(
                set(fieldnames)
                - set(harmonization_config.mapped_labels)
                - _known_metadata_columns(harmonization_config)
            )
        )

        manifest_rows: list[ExternalValidationManifestRow] = []
        label_table_rows: list[ExternalValidationLabelTableRow] = []
        seen_sample_ids: set[str] = set()

        for row_index, row in enumerate(reader):
            if max_rows is not None and len(manifest_rows) == max_rows:
                break

            manifest_row = _build_external_manifest_row(
                row,
                dataset_name=normalized_dataset_name,
                config=harmonization_config,
            )
            if manifest_row.sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate sample_id found in external metadata row {row_index + 2}: "
                    f"{manifest_row.sample_id!r}"
                )
            seen_sample_ids.add(manifest_row.sample_id)
            manifest_rows.append(manifest_row)

            label_table_rows.append(
                _build_external_label_table_row(
                    row,
                    manifest_row=manifest_row,
                    label_names=expected_label_names,
                    harmonization_config=harmonization_config,
                    harmonization_reference=protocol_config.label_harmonization_reference,
                )
            )

    if not manifest_rows:
        raise ValueError("External metadata CSV must contain at least one data row")

    overlapping_internal_patient_ids: tuple[str, ...] = ()
    if internal_manifest_path is not None:
        overlapping_internal_patient_ids = detect_patient_overlap_with_manifest(
            external_rows=manifest_rows,
            internal_manifest_path=internal_manifest_path,
        )
        if overlapping_internal_patient_ids:
            raise ValueError(
                "External/internal patient overlap detected: "
                + ", ".join(overlapping_internal_patient_ids[:5])
            )

    return ExternalValidationPreparation(
        protocol_config=protocol_config,
        harmonization_config=harmonization_config,
        label_names=expected_label_names,
        manifest_rows=tuple(manifest_rows),
        label_table_rows=tuple(label_table_rows),
        excluded_source_labels=excluded_source_labels,
        overlapping_internal_patient_ids=overlapping_internal_patient_ids,
    )


def detect_patient_overlap_with_manifest(
    *,
    external_rows: Sequence[ExternalValidationManifestRow],
    internal_manifest_path: str | Path,
) -> tuple[str, ...]:
    """Return sorted overlapping patient IDs across external and internal manifests."""

    manifest_path = Path(internal_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Internal/dev manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Internal/dev manifest CSV must include a header row")
        if "patient_id" not in reader.fieldnames:
            raise ValueError(
                "Internal/dev manifest CSV must contain a 'patient_id' column for overlap checks"
            )

        internal_patient_ids = {
            patient_id
            for row in reader
            if (patient_id := _optional_string(row.get("patient_id"))) is not None
        }

    external_patient_ids = {
        row.patient_id for row in external_rows if row.patient_id is not None
    }
    return tuple(sorted(external_patient_ids & internal_patient_ids))


def write_external_validation_manifest_csv(
    rows: Sequence[ExternalValidationManifestRow],
    output_path: str | Path,
) -> Path:
    """Write a validated external manifest CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "sample_id",
                "dataset_name",
                "patient_id",
                "study_id",
                "split",
                "view_position",
                "image_path",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))
    return path


def write_external_validation_label_table_csv(
    rows: Sequence[ExternalValidationLabelTableRow],
    output_path: str | Path,
    *,
    label_names: Sequence[str],
) -> Path:
    """Write a Phase 6-compatible external label-table CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "sample_id",
                "dataset_name",
                "patient_id",
                "study_id",
                "split",
                "view_position",
                "harmonization_reference",
                *label_names,
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_flat_row(label_names=label_names))
    return path


def _build_external_manifest_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    config: ExternalLabelHarmonizationConfig,
) -> ExternalValidationManifestRow:
    sample_id = _first_required_string(row, config.sample_id_columns, "sample ID")
    image_path = _first_required_string(row, config.image_path_columns, "image path")

    return ExternalValidationManifestRow(
        sample_id=sample_id,
        dataset_name=dataset_name,
        image_path=image_path,
        patient_id=_first_optional_string(row, config.patient_id_columns),
        study_id=_first_optional_string(row, config.study_id_columns),
        split=_first_optional_string(row, config.split_columns),
        view_position=_first_optional_string(row, config.view_position_columns),
    )


def _build_external_label_table_row(
    row: Mapping[str, Any],
    *,
    manifest_row: ExternalValidationManifestRow,
    label_names: Sequence[str],
    harmonization_config: ExternalLabelHarmonizationConfig,
    harmonization_reference: str,
) -> ExternalValidationLabelTableRow:
    source_by_project = {
        project_label: source_label
        for source_label, project_label in harmonization_config.mapped_labels.items()
    }
    labels: dict[str, float | None] = {}
    for label_name in label_names:
        source_label = source_by_project.get(label_name)
        if source_label is None:
            raise ValueError(
                f"No source-label mapping configured for project label {label_name!r}"
            )
        labels[label_name] = _normalize_external_binary_label_value(row.get(source_label))

    return ExternalValidationLabelTableRow(
        sample_id=manifest_row.sample_id,
        dataset_name=manifest_row.dataset_name,
        patient_id=manifest_row.patient_id,
        study_id=manifest_row.study_id,
        split=manifest_row.split,
        view_position=manifest_row.view_position,
        harmonization_reference=harmonization_reference,
        labels=labels,
    )


def _validate_required_harmonization_columns(
    *,
    fieldnames: Sequence[str],
    config: ExternalLabelHarmonizationConfig,
) -> None:
    available_columns = set(fieldnames)
    missing_label_columns = [
        source_label
        for source_label in config.mapped_labels
        if source_label not in available_columns
    ]
    if missing_label_columns:
        raise ValueError(
            "External metadata CSV is missing required mapped label column(s): "
            + ", ".join(missing_label_columns)
        )

    if not available_columns & set(config.sample_id_columns):
        raise ValueError(
            "External metadata CSV must contain at least one sample identifier column: "
            + ", ".join(config.sample_id_columns)
        )
    if not available_columns & set(config.image_path_columns):
        raise ValueError(
            "External metadata CSV must contain at least one image-path column: "
            + ", ".join(config.image_path_columns)
        )


def _known_metadata_columns(config: ExternalLabelHarmonizationConfig) -> set[str]:
    return {
        *config.sample_id_columns,
        *config.image_path_columns,
        *config.patient_id_columns,
        *config.study_id_columns,
        *config.split_columns,
        *config.view_position_columns,
        *config.known_metadata_columns,
    }


def _first_required_string(
    row: Mapping[str, Any],
    candidate_keys: Sequence[str],
    field_description: str,
) -> str:
    value = _first_optional_string(row, candidate_keys)
    if value is not None:
        return value
    raise ValueError(
        f"Missing required {field_description} column. Tried: " + ", ".join(candidate_keys)
    )


def _first_optional_string(
    row: Mapping[str, Any],
    candidate_keys: Sequence[str],
) -> str | None:
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


def _normalize_external_binary_label_value(value: Any) -> float | None:
    """Normalize external candidate labels into 1.0, 0.0, or None."""

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
        raise ValueError(f"Invalid external binary label value: {value!r}") from exc

    if numeric_value == 1.0:
        return 1.0
    if numeric_value == 0.0:
        return 0.0

    raise ValueError(
        "External validation labels must be binary 1/0 or missing. "
        f"Received: {value!r}"
    )
