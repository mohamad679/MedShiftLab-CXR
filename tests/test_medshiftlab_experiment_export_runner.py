"""Tests for the MedShiftLab-CXR file-exporting experiment runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from medshiftlab.experiments import (
    ExportedExperimentResult,
    InMemoryExperimentConfig,
    run_and_export_evaluation_experiment,
)
from medshiftlab.models import MockCXRModelAdapter


LABELS = ("Atelectasis", "Cardiomegaly")


def _config() -> InMemoryExperimentConfig:
    return InMemoryExperimentConfig(
        dataset_name="CheXpert",
        labels=LABELS,
        split="validation",
        uncertainty_strategy="U-ignore",
        n_bins=2,
    )


def _records() -> list[dict[str, object]]:
    return [
        {
            "image_id": "img001",
            "image_path": "images/img001.png",
            "dataset_name": "CheXpert",
            "labels": {"Atelectasis": 0.0, "Cardiomegaly": 1.0},
        },
        {
            "image_id": "img002",
            "image_path": "images/img002.png",
            "dataset_name": "CheXpert",
            "labels": {"Atelectasis": 1.0, "Cardiomegaly": 0.0},
        },
    ]


def _adapter() -> MockCXRModelAdapter:
    return MockCXRModelAdapter("mock-cxr", LABELS, default_score=0.25)


def _run(output_dir: str | Path) -> ExportedExperimentResult:
    return run_and_export_evaluation_experiment(
        records=_records(),
        adapter=_adapter(),
        config=_config(),
        output_dir=output_dir,
    )


def test_exported_runner_returns_exported_experiment_result(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert isinstance(result, ExportedExperimentResult)
    assert result.output_dir == tmp_path
    assert result.experiment_result.n_records == 2


def test_exported_runner_writes_json_and_csv_files(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.output_paths["json"].is_file()
    assert result.output_paths["label_metrics_csv"].is_file()


def test_output_paths_use_stable_export_keys(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert set(result.output_paths) == {"json", "label_metrics_csv"}


def test_exported_result_rejects_empty_output_paths(tmp_path: Path) -> None:
    result = _run(tmp_path)

    with pytest.raises(ValueError):
        ExportedExperimentResult(
            experiment_result=result.experiment_result,
            output_dir=tmp_path,
            output_paths={},
        )


def test_output_directory_accepts_string_path(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "string-output"))

    assert isinstance(result.output_dir, Path)
    assert result.output_dir == tmp_path / "string-output"
    assert all(path.is_file() for path in result.output_paths.values())


def test_custom_export_filenames_work(tmp_path: Path) -> None:
    result = run_and_export_evaluation_experiment(
        records=_records(),
        adapter=_adapter(),
        config=_config(),
        output_dir=tmp_path,
        json_filename="custom-report.json",
        csv_filename="custom-labels.csv",
    )

    assert result.output_paths == {
        "json": tmp_path / "custom-report.json",
        "label_metrics_csv": tmp_path / "custom-labels.csv",
    }
    assert all(path.is_file() for path in result.output_paths.values())


def test_exported_json_contains_dataset_and_model_name(tmp_path: Path) -> None:
    result = _run(tmp_path)

    payload = json.loads(result.output_paths["json"].read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset_name"] == "CheXpert"
    assert payload["metadata"]["model_name"] == "mock-cxr"


def test_exported_csv_contains_one_row_per_label(tmp_path: Path) -> None:
    result = _run(tmp_path)

    with result.output_paths["label_metrics_csv"].open(
        encoding="utf-8", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == len(LABELS)
    assert [row["label_name"] for row in rows] == list(LABELS)


def test_exported_metadata_matches_in_memory_report(tmp_path: Path) -> None:
    result = _run(tmp_path)

    payload = json.loads(result.output_paths["json"].read_text(encoding="utf-8"))
    expected_metadata = result.experiment_result.report.metadata.model_dump(mode="json")
    assert payload["metadata"] == expected_metadata


def test_empty_records_are_rejected_before_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "should-not-exist"

    with pytest.raises(ValueError, match="records must not be empty"):
        run_and_export_evaluation_experiment(
            records=[],
            adapter=_adapter(),
            config=_config(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()
