"""Dataset registry and local-only path configuration.

The tracked example configuration intentionally contains no dataset paths.
Users may copy it to the ignored local configuration filename and populate that
copy without exposing private filesystem locations in Git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EXAMPLE_LOCAL_PATHS_CONFIG = Path("configs/data/example_local_paths.yaml")
LOCAL_PATHS_CONFIG = Path("configs/data/local_paths.yaml")


class DatasetRegistryEntry(BaseModel):
    """Static metadata describing one supported dataset path contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: Literal["development_internal", "external_candidate"]
    root_path_field: str = Field(min_length=1)
    metadata_path_field: str | None
    image_directory_field: str | None
    external_validation_only: bool
    notes: str = Field(min_length=1)

    @property
    def required_path_fields(self) -> tuple[str, ...]:
        """Return the configured local path fields in stable order."""

        return tuple(
            field_name
            for field_name in (
                self.root_path_field,
                self.metadata_path_field,
                self.image_directory_field,
            )
            if field_name is not None
        )


DATASET_REGISTRY: dict[str, DatasetRegistryEntry] = {
    "chexpert": DatasetRegistryEntry(
        name="chexpert",
        display_name="CheXpert",
        role="development_internal",
        root_path_field="root_path",
        metadata_path_field="metadata_path",
        image_directory_field="image_directory",
        external_validation_only=False,
        notes=(
            "Development/internal protocol dataset. Authorized access is required; "
            "raw data must not be committed."
        ),
    ),
    "mimic_cxr_jpg": DatasetRegistryEntry(
        name="mimic_cxr_jpg",
        display_name="MIMIC-CXR-JPG",
        role="external_candidate",
        root_path_field="root_path",
        metadata_path_field="metadata_path",
        image_directory_field="image_directory",
        external_validation_only=True,
        notes=(
            "External-validation candidate with restricted access; raw data must not "
            "be committed or used for protocol tuning."
        ),
    ),
    "vindr_cxr": DatasetRegistryEntry(
        name="vindr_cxr",
        display_name="VinDr-CXR",
        role="external_candidate",
        root_path_field="root_path",
        metadata_path_field="metadata_path",
        image_directory_field="image_directory",
        external_validation_only=True,
        notes=(
            "External-validation candidate requiring authorized local access; raw data "
            "must not be committed or used for protocol tuning."
        ),
    ),
}

SUPPORTED_DATASET_NAMES = tuple(DATASET_REGISTRY)


class DatasetLocalPaths(BaseModel):
    """Path placeholders for one dataset; null values mean not configured."""

    model_config = ConfigDict(extra="forbid")

    root_path: Path | None
    metadata_path: Path | None
    image_directory: Path | None

    @field_validator("root_path", "metadata_path", "image_directory", mode="before")
    @classmethod
    def _reject_blank_paths(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("local path values must be non-empty strings or null")
        return value


class LocalDataConfig(BaseModel):
    """Validated local dataset-path configuration."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    datasets: dict[str, DatasetLocalPaths]
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _validate_dataset_sections(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        datasets = value.get("datasets")
        if not isinstance(datasets, dict):
            return value

        required_fields = set(DatasetLocalPaths.model_fields)
        for dataset_name, path_values in datasets.items():
            entry = get_dataset_registry_entry(dataset_name)
            if dataset_name != entry.name:
                raise ValueError(
                    f"Dataset configuration key must use canonical name '{entry.name}'"
                )
            if not isinstance(path_values, dict):
                raise ValueError(
                    f"Local path configuration for dataset '{entry.name}' must be a mapping"
                )

            missing_fields = required_fields.difference(path_values)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Local path configuration for dataset '{entry.name}' is missing "
                    f"required field(s): {missing}"
                )

        return value


def get_dataset_registry_entry(dataset_name: str) -> DatasetRegistryEntry:
    """Return one registry entry or raise a clear error for an unknown name."""

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")

    normalized_name = dataset_name.strip()
    try:
        return DATASET_REGISTRY[normalized_name]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_DATASET_NAMES)
        raise ValueError(
            f"Unknown dataset '{normalized_name}'. Supported datasets: {supported}"
        ) from error


def load_local_data_config(path: str | Path) -> LocalDataConfig:
    """Load a tracked example or ignored real local path configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Local data configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Local data configuration YAML must contain a top-level mapping")

    config = LocalDataConfig.model_validate(raw_data)
    config.source_path = config_path
    return config


def load_example_local_data_config() -> LocalDataConfig:
    """Load the repository's path-free example configuration."""

    repo_root = Path(__file__).resolve().parents[3]
    return load_local_data_config(repo_root / EXAMPLE_LOCAL_PATHS_CONFIG)


def require_local_dataset_paths(
    config: LocalDataConfig,
    dataset_name: str,
) -> dict[str, Path]:
    """Return configured paths, rejecting absent datasets and null path values."""

    entry = get_dataset_registry_entry(dataset_name)
    path_config = config.datasets.get(entry.name)
    source = f" in {config.source_path}" if config.source_path is not None else ""

    if path_config is None:
        raise ValueError(f"Dataset '{entry.name}' is not configured{source}")

    missing_fields = [
        field_name
        for field_name in entry.required_path_fields
        if getattr(path_config, field_name) is None
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Missing required local path(s) for dataset '{entry.name}': {missing}. "
            f"Populate the ignored {LOCAL_PATHS_CONFIG} file; do not commit private paths."
        )

    return {
        field_name: path
        for field_name in entry.required_path_fields
        if (path := getattr(path_config, field_name)) is not None
    }
