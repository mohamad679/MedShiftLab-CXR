"""Focused tests for Phase 7 CheXpert internal-protocol scaffolding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from medshiftlab.data import (
    CheXpertSplitConfig,
    assign_patient_level_splits,
    build_split_manifest_rows,
    load_chexpert_metadata_csv,
    materialize_chexpert_label_table_rows,
    prepare_chexpert_internal_protocol,
    write_chexpert_label_table_csv,
)
from medshiftlab.experiments import (
    PredictionEvaluationConfig,
    run_prediction_batch_evaluation_from_files,
)
from medshiftlab.labels import load_default_label_ontology
from medshiftlab.models import PREDICTION_SCHEMA_VERSION, PredictionBatch, PredictionRecord
from medshiftlab.reporting import write_prediction_batch_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    return environment


def _write_chexpert_like_metadata(csv_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "Sex": "Female",
                "Age": 65,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "PA",
                "Atelectasis": -1,
                "Cardiomegaly": 1,
                "Pleural Effusion": 0,
                "Pneumonia": 0,
                "Pneumothorax": None,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00001/study2/view1_frontal.jpg",
                "Sex": "Female",
                "Age": 65,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "PA",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
                "Pleural Effusion": 0,
                "Pneumonia": 1,
                "Pneumothorax": 0,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                "Sex": "Male",
                "Age": 72,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "AP",
                "Atelectasis": 0,
                "Cardiomegaly": -1,
                "Pleural Effusion": 1,
                "Pneumonia": 0,
                "Pneumothorax": 0,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00003/study1/view1_frontal.jpg",
                "Sex": "Female",
                "Age": 48,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "PA",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
                "Pleural Effusion": 0,
                "Pneumonia": -1,
                "Pneumothorax": 1,
                "No Finding": 0,
            },
        ]
    ).to_csv(csv_path, index=False)


def _prediction_batch(sample_ids: list[str]) -> PredictionBatch:
    records = [
        PredictionRecord(
            sample_id=sample_id,
            dataset_name="chexpert",
            model_name="mock-cxr",
            label_names=(
                "Atelectasis",
                "Cardiomegaly",
                "Pleural Effusion",
                "Pneumonia",
                "Pneumothorax",
                "No Finding",
            ),
            probabilities=(0.8, 0.2, 0.1, 0.7, 0.4, 0.05),
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
        label_names=records[0].label_names,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        uncertainty_strategy="U-zero",
        run_metadata={"manual_only": True},
    )


def test_patient_level_split_has_no_leakage_and_keeps_duplicate_patient_rows_together(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    ontology = load_default_label_ontology()
    records = load_chexpert_metadata_csv(metadata_csv, ontology, "U-ignore")

    split_records = assign_patient_level_splits(
        records,
        config=CheXpertSplitConfig(
            split_names=("train", "validation", "test"),
            split_fractions=(0.5, 0.25, 0.25),
            seed=17,
        ),
    )
    manifest_rows = build_split_manifest_rows(split_records)

    patient_to_split = {row.patient_id: row.split for row in manifest_rows}
    assert len(patient_to_split) == 3
    patient00001_splits = {
        row.split for row in manifest_rows if row.patient_id == "patient00001"
    }
    assert patient00001_splits == {patient_to_split["patient00001"]}


def test_patient_level_split_is_deterministic_for_seed(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    ontology = load_default_label_ontology()
    records = load_chexpert_metadata_csv(metadata_csv, ontology, "U-ignore")
    config = CheXpertSplitConfig(seed=23)

    first = build_split_manifest_rows(assign_patient_level_splits(records, config=config))
    second = build_split_manifest_rows(assign_patient_level_splits(records, config=config))

    assert [
        (row.sample_id, row.split) for row in first
    ] == [
        (row.sample_id, row.split) for row in second
    ]


def test_patient_level_split_rejects_missing_patient_id(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    ontology = load_default_label_ontology()
    records = load_chexpert_metadata_csv(metadata_csv, ontology, "U-ignore")
    missing_patient_record = records[0].model_copy(
        update={"patient_id": None, "image_id": "local/view1.jpg"}
    )

    with pytest.raises(ValueError, match="requires patient_id"):
        assign_patient_level_splits([missing_patient_record, *records[1:]])


@pytest.mark.parametrize(
    ("strategy", "expected_value"),
    [
        ("U-ignore", None),
        ("U-zero", 0.0),
        ("U-one", 1.0),
        ("U-soft", 0.5),
    ],
)
def test_uncertainty_strategy_materialization_handles_uncertain_labels(
    tmp_path: Path,
    strategy: str,
    expected_value: float | None,
) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    ontology = load_default_label_ontology()
    base_records = load_chexpert_metadata_csv(metadata_csv, ontology, "U-ignore")
    split_manifest = build_split_manifest_rows(assign_patient_level_splits(base_records))
    split_by_sample_id = {row.sample_id: row.split for row in split_manifest}

    strategy_records = load_chexpert_metadata_csv(metadata_csv, ontology, strategy)
    label_rows = materialize_chexpert_label_table_rows(
        strategy_records,
        uncertainty_strategy=strategy,
        split_by_sample_id=split_by_sample_id,
    )

    assert label_rows[0].uncertainty_strategy == strategy
    assert label_rows[0].labels["Atelectasis"] == expected_value


def test_generated_label_tables_are_phase6_evaluation_compatible(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    preparation = prepare_chexpert_internal_protocol(
        metadata_csv,
        uncertainty_strategies=("U-zero",),
        max_rows=2,
    )
    label_rows = preparation.label_table_rows_by_strategy["U-zero"]
    labels_csv = write_chexpert_label_table_csv(
        label_rows,
        tmp_path / "chexpert_labels_u_zero.csv",
        label_names=preparation.label_names,
    )
    predictions_json = write_prediction_batch_json(
        _prediction_batch([row.sample_id for row in label_rows]),
        tmp_path / "predictions.json",
    )

    result, _ = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_json,
        labels_csv_path=labels_csv,
        config=PredictionEvaluationConfig(limit=2, threshold=0.5, n_bins=2),
    )

    assert result.accounting.evaluated_records == 2
    assert result.report.metadata.dataset_name == "chexpert"


def test_split_manifest_rejects_absolute_image_paths(tmp_path: Path) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    _write_chexpert_like_metadata(metadata_csv)
    ontology = load_default_label_ontology()
    record = load_chexpert_metadata_csv(metadata_csv, ontology, "U-ignore", max_rows=1)[0]
    absolute_path_record = record.model_copy(update={"image_path": "/private/secret.jpg"})

    with pytest.raises(ValueError, match="relative"):
        build_split_manifest_rows({"train": [absolute_path_record]})


def test_prepare_chexpert_internal_protocol_script_works_without_private_paths(
    tmp_path: Path,
) -> None:
    metadata_csv = tmp_path / "chexpert_metadata.csv"
    output_dir = tmp_path / "local_private_results"
    _write_chexpert_like_metadata(metadata_csv)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_chexpert_internal_protocol.py",
            "--metadata-csv",
            str(metadata_csv),
            "--output-dir",
            str(output_dir),
            "--limit",
            "4",
            "--write-split-manifest",
            "--write-label-tables",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
    )

    payload = json.loads(result.stdout)
    assert payload["dataset_name"] == "chexpert"
    assert payload["processed_records"] == 4
    assert payload["phase6_evaluation_input_compatible"] is True
    assert str(tmp_path) not in result.stdout
    assert payload["outputs_written"]["split_manifest"] == "chexpert_internal_split_manifest.csv"
    assert set(payload["outputs_written"]["label_tables"]) == {
        "U-ignore",
        "U-zero",
        "U-one",
        "U-soft",
    }

    split_manifest = output_dir / "chexpert_internal_split_manifest.csv"
    assert split_manifest.exists()
    assert str(tmp_path) not in split_manifest.read_text(encoding="utf-8")

    for filename in payload["outputs_written"]["label_tables"].values():
        label_table_path = output_dir / filename
        assert label_table_path.exists()
        assert str(tmp_path) not in label_table_path.read_text(encoding="utf-8")
