"""Focused tests for Phase 8 external-validation setup scaffolding."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from medshiftlab.data import (
    default_external_label_harmonization_config_path,
    default_external_protocol_config_path,
    detect_patient_overlap_with_manifest,
    load_external_label_harmonization_config,
    load_external_validation_protocol_config,
    prepare_external_validation_protocol,
    write_external_validation_label_table_csv,
)
from medshiftlab.experiments import (
    PredictionEvaluationConfig,
    run_prediction_batch_evaluation_from_files,
)
from medshiftlab.models import PREDICTION_SCHEMA_VERSION, PredictionBatch, PredictionRecord
from medshiftlab.reporting import write_prediction_batch_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    return environment


def _write_mimic_metadata(csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dicom_id",
                "subject_id",
                "study_id",
                "path",
                "split",
                "ViewPosition",
                "Atelectasis",
                "Cardiomegaly",
                "Pleural Effusion",
                "Pneumonia",
                "Pneumothorax",
                "No Finding",
                "Support Devices",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "dicom_id": "mimic001",
                    "subject_id": "pat001",
                    "study_id": "study001",
                    "path": "files/pat001/study001/img1.jpg",
                    "split": "test",
                    "ViewPosition": "PA",
                    "Atelectasis": 1,
                    "Cardiomegaly": 0,
                    "Pleural Effusion": 0,
                    "Pneumonia": 1,
                    "Pneumothorax": 0,
                    "No Finding": 0,
                    "Support Devices": 1,
                },
                {
                    "dicom_id": "mimic002",
                    "subject_id": "pat002",
                    "study_id": "study002",
                    "path": "files/pat002/study002/img2.jpg",
                    "split": "test",
                    "ViewPosition": "AP",
                    "Atelectasis": 0,
                    "Cardiomegaly": 1,
                    "Pleural Effusion": 1,
                    "Pneumonia": 0,
                    "Pneumothorax": 0,
                    "No Finding": 0,
                    "Support Devices": 0,
                },
            ]
        )


def _write_vindr_metadata(csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "ImageID",
                "PatientID",
                "StudyID",
                "Path",
                "Split",
                "ViewPosition",
                "Atelectasis",
                "Cardiomegaly",
                "Pleural effusion",
                "Pneumonia",
                "Pneumothorax",
                "No finding",
                "Aortic enlargement",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "vindr001",
                    "PatientID": "vpat001",
                    "StudyID": "vstudy001",
                    "Path": "images/vindr001.png",
                    "Split": "test",
                    "ViewPosition": "PA",
                    "Atelectasis": 1,
                    "Cardiomegaly": 0,
                    "Pleural effusion": 1,
                    "Pneumonia": 0,
                    "Pneumothorax": 0,
                    "No finding": 0,
                    "Aortic enlargement": 1,
                }
            ]
        )


def _write_internal_manifest(csv_path: Path, patient_ids: list[str]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_id",
                "dataset_name",
                "patient_id",
                "split",
                "image_path",
            ],
        )
        writer.writeheader()
        for index, patient_id in enumerate(patient_ids, start=1):
            writer.writerow(
                {
                    "sample_id": f"internal{index:03d}",
                    "dataset_name": "chexpert",
                    "patient_id": patient_id,
                    "split": "test",
                    "image_path": f"train/{patient_id}/img{index}.jpg",
                }
            )


def _prediction_batch(dataset_name: str, sample_ids: list[str]) -> PredictionBatch:
    label_names = (
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
        "No Finding",
    )
    records = [
        PredictionRecord(
            sample_id=sample_id,
            dataset_name=dataset_name,
            model_name="mock-cxr",
            label_names=label_names,
            probabilities=(0.8, 0.2, 0.3, 0.7, 0.1, 0.05),
        )
        for sample_id in sample_ids
    ]
    return PredictionBatch(
        schema_version=PREDICTION_SCHEMA_VERSION,
        model_name="mock-cxr",
        model_version="mock-cxr:v1",
        adapter_name="mock-cxr-adapter",
        preprocessing_version="preprocess-v1",
        preprocessing_config={"output_mode": "grayscale"},
        records=records,
        label_names=label_names,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_metadata={"manual_only": True},
    )


def test_mimic_cxr_jpg_harmonization_works_on_synthetic_labels(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "mimic_metadata.csv"
    _write_mimic_metadata(metadata_csv)

    preparation = prepare_external_validation_protocol(
        "mimic_cxr_jpg",
        metadata_csv,
        max_rows=2,
    )

    assert len(preparation.manifest_rows) == 2
    assert preparation.manifest_rows[0].dataset_name == "mimic_cxr_jpg"
    assert preparation.label_table_rows[0].labels["Atelectasis"] == 1.0
    assert preparation.label_table_rows[1].labels["Cardiomegaly"] == 1.0


def test_vindr_cxr_harmonization_works_on_synthetic_labels(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "vindr_metadata.csv"
    _write_vindr_metadata(metadata_csv)

    preparation = prepare_external_validation_protocol(
        "vindr_cxr",
        metadata_csv,
        max_rows=1,
    )

    assert len(preparation.manifest_rows) == 1
    assert preparation.label_table_rows[0].labels["Pleural Effusion"] == 1.0
    assert preparation.excluded_source_labels == ("Aortic enlargement",)


def test_missing_mapped_label_columns_fail_clearly(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "mimic_missing.csv"
    _write_mimic_metadata(metadata_csv)
    rows = list(csv.DictReader(metadata_csv.open(encoding="utf-8", newline="")))
    with metadata_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[field for field in rows[0] if field != "Pneumonia"],
        )
        writer.writeheader()
        for row in rows:
            row.pop("Pneumonia", None)
            writer.writerow(row)

    with pytest.raises(ValueError, match="missing required mapped label column"):
        prepare_external_validation_protocol("mimic_cxr_jpg", metadata_csv)


def test_duplicate_sample_ids_fail_clearly(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "mimic_duplicates.csv"
    _write_mimic_metadata(metadata_csv)
    rows = list(csv.DictReader(metadata_csv.open(encoding="utf-8", newline="")))
    rows[1]["dicom_id"] = rows[0]["dicom_id"]
    with metadata_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        prepare_external_validation_protocol("mimic_cxr_jpg", metadata_csv)


@pytest.mark.parametrize(
    ("bad_path", "pattern"),
    [
        ("/absolute/path.jpg", "relative"),
        ("../escape/path.jpg", "remain inside"),
    ],
)
def test_bad_image_paths_fail_clearly(
    tmp_path: Path,
    bad_path: str,
    pattern: str,
) -> None:
    metadata_csv = tmp_path / "mimic_bad_paths.csv"
    _write_mimic_metadata(metadata_csv)
    rows = list(csv.DictReader(metadata_csv.open(encoding="utf-8", newline="")))
    rows[0]["path"] = bad_path
    with metadata_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=pattern):
        prepare_external_validation_protocol("mimic_cxr_jpg", metadata_csv)


def test_external_only_configs_prohibit_tuning_fields(tmp_path: Path) -> None:
    protocol = load_external_validation_protocol_config(
        default_external_protocol_config_path("mimic_cxr_jpg")
    )
    assert protocol.allow_threshold_tuning is False
    assert protocol.allow_hyperparameter_tuning is False
    assert protocol.allow_model_selection is False
    assert protocol.allow_protocol_edit_after_results is False

    bad_protocol_path = tmp_path / "bad_external_protocol.yaml"
    bad_protocol_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "dataset": "mimic_cxr_jpg",
                "role": "external_validation_candidate",
                "external_validation_only": True,
                "allow_threshold_tuning": True,
                "allow_hyperparameter_tuning": False,
                "allow_model_selection": False,
                "allow_protocol_edit_after_results": False,
                "label_set_reference": "configs/labels/cxr_common_labels.yaml",
                "label_harmonization_reference": "configs/protocol/harmonization/mimic_cxr_jpg_labels.yaml",
                "prediction_schema_version": "medshiftlab.prediction.v1",
                "evaluation_metrics_reference": "docs/medshiftlab/research_protocol.md#5-primary-metrics-and-uncertainty-estimates",
                "safe_default_limit": 256,
                "local_private_output_dir": "results/local_private_runs/bad/",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_external_validation_protocol_config(bad_protocol_path)


def test_optional_internal_external_patient_overlap_check_works(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "mimic_metadata.csv"
    internal_manifest_csv = tmp_path / "internal_manifest.csv"
    _write_mimic_metadata(metadata_csv)
    _write_internal_manifest(internal_manifest_csv, ["pat001", "other"])

    preparation = prepare_external_validation_protocol(
        "mimic_cxr_jpg",
        metadata_csv,
        max_rows=2,
    )
    overlaps = detect_patient_overlap_with_manifest(
        external_rows=preparation.manifest_rows,
        internal_manifest_path=internal_manifest_csv,
    )
    assert overlaps == ("pat001",)

    with pytest.raises(ValueError, match="patient overlap"):
        prepare_external_validation_protocol(
            "mimic_cxr_jpg",
            metadata_csv,
            max_rows=2,
            internal_manifest_path=internal_manifest_csv,
        )


def test_generated_label_tables_are_phase6_evaluation_compatible(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "mimic_metadata.csv"
    _write_mimic_metadata(metadata_csv)
    preparation = prepare_external_validation_protocol(
        "mimic_cxr_jpg",
        metadata_csv,
        max_rows=2,
    )
    labels_csv = write_external_validation_label_table_csv(
        preparation.label_table_rows,
        tmp_path / "mimic_external_labels.csv",
        label_names=preparation.label_names,
    )
    predictions_json = write_prediction_batch_json(
        _prediction_batch(
            "mimic_cxr_jpg",
            [row.sample_id for row in preparation.label_table_rows],
        ),
        tmp_path / "predictions.json",
    )

    result, _ = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_json,
        labels_csv_path=labels_csv,
        config=PredictionEvaluationConfig(limit=2, threshold=0.5, n_bins=2),
    )

    assert result.accounting.evaluated_records == 2
    assert result.report.metadata.dataset_name == "mimic_cxr_jpg"


def test_script_works_on_synthetic_metadata_only_and_avoids_private_paths(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "vindr_metadata.csv"
    output_dir = tmp_path / "local_private_results"
    _write_vindr_metadata(metadata_csv)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_external_validation_protocol.py",
            "--dataset",
            "vindr_cxr",
            "--metadata-csv",
            str(metadata_csv),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--write-manifest",
            "--write-label-table",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
    )

    payload = json.loads(result.stdout)
    assert payload["dataset_name"] == "vindr_cxr"
    assert payload["processed_records"] == 1
    assert payload["phase6_evaluation_input_compatible"] is True
    assert payload["excluded_source_labels"] == ["Aortic enlargement"]
    assert str(tmp_path) not in result.stdout

    manifest_path = output_dir / "vindr_cxr_external_manifest.csv"
    label_table_path = output_dir / "vindr_cxr_external_labels.csv"
    assert manifest_path.exists()
    assert label_table_path.exists()
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in label_table_path.read_text(encoding="utf-8")


def test_default_external_harmonization_config_loads() -> None:
    config = load_external_label_harmonization_config(
        default_external_label_harmonization_config_path("vindr_cxr")
    )

    assert config.dataset == "vindr_cxr"
    assert config.mapped_labels["Pleural effusion"] == "Pleural Effusion"
