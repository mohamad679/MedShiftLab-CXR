from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def _write_reference_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = pd.DataFrame(
        {
            "sample_id": ["vindr_img001", "vindr_img002"],
            "dataset_name": ["vindr_cxr", "vindr_cxr"],
            "image_path": ["img001.jpg", "img002.jpg"],
            "source_image_id": ["img001", "img002"],
            "image_found": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["vindr_img001", "vindr_img002"],
            "source_image_id": ["img001", "img002"],
            "Atelectasis": [1.0, 0.0],
            "Cardiomegaly": [0.0, 1.0],
            "Pleural Effusion": [0.0, 0.0],
            "Pneumonia": [0.0, 0.0],
            "Pneumothorax": [0.0, 0.0],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    labels_path = tmp_path / "labels.csv"
    manifest.to_csv(manifest_path, index=False)
    labels.to_csv(labels_path, index=False)
    return manifest_path, labels_path


def _write_predictions(tmp_path: Path, *, rows: list[dict[str, object]] | None = None) -> Path:
    rows = rows or [
        {
            "sample_id": "vindr_img001",
            "Atelectasis": 0.8,
            "Cardiomegaly": 0.1,
            "Pleural Effusion": 0.2,
            "Pneumonia": 0.0,
            "Pneumothorax": 0.05,
        },
        {
            "sample_id": "vindr_img002",
            "Atelectasis": 0.1,
            "Cardiomegaly": 0.7,
            "Pleural Effusion": 0.3,
            "Pneumonia": 0.0,
            "Pneumothorax": 0.02,
        },
    ]
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    return predictions_path


def _run_validator(
    tmp_path: Path,
    *,
    predictions_path: Path,
    manifest_path: Path,
    labels_path: Path,
    allow_subset: bool = False,
) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo / "scripts/validate_vindr_prediction_schema.py"),
        "--predictions-csv",
        str(predictions_path),
        "--manifest-csv",
        str(manifest_path),
        "--labels-csv",
        str(labels_path),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    if allow_subset:
        command.append("--allow-subset")

    return subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_vindr_prediction_schema_validator_accepts_valid_predictions(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_reference_inputs(tmp_path)
    predictions_path = _write_predictions(tmp_path)

    result = _run_validator(
        tmp_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert result.returncode == 0
    summary_path = tmp_path / "out" / "vindr_prediction_schema_validation_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "medshiftlab.vindr_prediction_schema_validation.v1"
    assert summary["claim_level"]["model_inference_completed"] is False
    assert summary["claim_level"]["metrics_completed"] is False
    assert summary["prediction_summary"]["n_prediction_rows"] == 2
    assert summary["outputs"]["metrics_json"] is None


def test_vindr_prediction_schema_validator_rejects_out_of_range_values(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_reference_inputs(tmp_path)
    predictions_path = _write_predictions(
        tmp_path,
        rows=[
            {
                "sample_id": "vindr_img001",
                "Atelectasis": 1.2,
                "Cardiomegaly": 0.1,
                "Pleural Effusion": 0.2,
                "Pneumonia": 0.0,
                "Pneumothorax": 0.05,
            },
            {
                "sample_id": "vindr_img002",
                "Atelectasis": 0.1,
                "Cardiomegaly": 0.7,
                "Pleural Effusion": 0.3,
                "Pneumonia": 0.0,
                "Pneumothorax": 0.02,
            },
        ],
    )

    result = _run_validator(
        tmp_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert result.returncode != 0
    assert "within [0, 1]" in result.stdout


def test_vindr_prediction_schema_validator_rejects_missing_required_columns(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_reference_inputs(tmp_path)
    predictions_path = _write_predictions(tmp_path)
    predictions = pd.read_csv(predictions_path).drop(columns=["Pneumothorax"])
    predictions.to_csv(predictions_path, index=False)

    result = _run_validator(
        tmp_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert result.returncode != 0
    assert "missing required columns" in result.stdout


def test_vindr_prediction_schema_validator_rejects_missing_samples_by_default(
    tmp_path: Path,
) -> None:
    manifest_path, labels_path = _write_reference_inputs(tmp_path)
    predictions_path = _write_predictions(
        tmp_path,
        rows=[
            {
                "sample_id": "vindr_img001",
                "Atelectasis": 0.8,
                "Cardiomegaly": 0.1,
                "Pleural Effusion": 0.2,
                "Pneumonia": 0.0,
                "Pneumothorax": 0.05,
            }
        ],
    )

    result = _run_validator(
        tmp_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
    )

    assert result.returncode != 0
    assert "missing expected samples" in result.stdout


def test_vindr_prediction_schema_validator_allows_subset_when_requested(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_reference_inputs(tmp_path)
    predictions_path = _write_predictions(
        tmp_path,
        rows=[
            {
                "sample_id": "vindr_img001",
                "Atelectasis": 0.8,
                "Cardiomegaly": 0.1,
                "Pleural Effusion": 0.2,
                "Pneumonia": 0.0,
                "Pneumothorax": 0.05,
            }
        ],
    )

    result = _run_validator(
        tmp_path,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
        allow_subset=True,
    )

    assert result.returncode == 0
