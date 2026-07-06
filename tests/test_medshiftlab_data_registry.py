"""Tests for the local-only dataset registry configuration boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from medshiftlab.data import (
    DATASET_REGISTRY,
    SUPPORTED_DATASET_NAMES,
    get_dataset_registry_entry,
    load_example_local_data_config,
    load_local_data_config,
    require_local_dataset_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "data" / "example_local_paths.yaml"
REAL_LOCAL_CONFIG = "configs/data/local_paths.yaml"


def test_supported_dataset_names_are_registered() -> None:
    assert SUPPORTED_DATASET_NAMES == (
        "chexpert",
        "mimic_cxr_jpg",
        "vindr_cxr",
    )
    assert set(DATASET_REGISTRY) == set(SUPPORTED_DATASET_NAMES)


def test_registry_records_dataset_roles_and_required_fields() -> None:
    chexpert = get_dataset_registry_entry("chexpert")
    mimic = get_dataset_registry_entry("mimic_cxr_jpg")
    vindr = get_dataset_registry_entry("vindr_cxr")

    assert chexpert.role == "development_internal"
    assert chexpert.external_validation_only is False
    assert mimic.role == "external_candidate"
    assert mimic.external_validation_only is True
    assert vindr.role == "external_candidate"
    assert vindr.external_validation_only is True

    for entry in (chexpert, mimic, vindr):
        assert entry.required_path_fields == (
            "root_path",
            "metadata_path",
            "image_directory",
        )
        assert "must not be committed" in entry.notes


def test_example_config_loads_with_null_path_placeholders() -> None:
    config = load_example_local_data_config()

    assert config.version == 1
    assert set(config.datasets) == set(SUPPORTED_DATASET_NAMES)
    assert config.source_path == EXAMPLE_CONFIG
    for paths in config.datasets.values():
        assert paths.root_path is None
        assert paths.metadata_path is None
        assert paths.image_directory is None


def test_example_config_contains_no_absolute_or_private_user_paths() -> None:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    path_values = [
        value
        for dataset_paths in payload["datasets"].values()
        for value in dataset_paths.values()
    ]

    assert path_values
    assert all(value is None for value in path_values)


def test_real_local_config_is_ignored_but_example_is_trackable() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", REAL_LOCAL_CONFIG],
        cwd=REPO_ROOT,
        check=False,
    )
    example = subprocess.run(
        ["git", "check-ignore", "--quiet", str(EXAMPLE_CONFIG.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
    )

    assert ignored.returncode == 0
    assert example.returncode == 1


def test_unknown_dataset_name_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown dataset 'unknown_dataset'"):
        get_dataset_registry_entry("unknown_dataset")


def test_unknown_dataset_in_config_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text(
        """version: 1
datasets:
  unknown_dataset:
    root_path: null
    metadata_path: null
    image_directory: null
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Unknown dataset 'unknown_dataset'"):
        load_local_data_config(config_path)


def test_missing_required_config_field_fails_clearly(tmp_path: Path) -> None:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text(
        """version: 1
datasets:
  chexpert:
    root_path: null
    metadata_path: null
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="missing required field.*image_directory"):
        load_local_data_config(config_path)


def test_null_required_paths_fail_when_dataset_is_requested() -> None:
    config = load_example_local_data_config()

    with pytest.raises(
        ValueError,
        match=(
            "Missing required local path.*chexpert.*root_path, metadata_path, "
            "image_directory"
        ),
    ):
        require_local_dataset_paths(config, "chexpert")


def test_absent_dataset_config_fails_when_dataset_is_requested(tmp_path: Path) -> None:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text("version: 1\ndatasets: {}\n", encoding="utf-8")
    config = load_local_data_config(config_path)

    with pytest.raises(ValueError, match="Dataset 'vindr_cxr' is not configured"):
        require_local_dataset_paths(config, "vindr_cxr")


def test_configured_paths_are_returned_without_accessing_data(tmp_path: Path) -> None:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text(
        """version: 1
datasets:
  chexpert:
    root_path: private-data/chexpert
    metadata_path: private-data/chexpert/train.csv
    image_directory: private-data/chexpert/images
""",
        encoding="utf-8",
    )

    config = load_local_data_config(config_path)
    paths = require_local_dataset_paths(config, "chexpert")

    assert paths == {
        "root_path": Path("private-data/chexpert"),
        "metadata_path": Path("private-data/chexpert/train.csv"),
        "image_directory": Path("private-data/chexpert/images"),
    }
