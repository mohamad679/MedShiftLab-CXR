from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
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


def _run_scaffold(tmp_path: Path, manifest_path: Path, labels_path: Path) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_vindr_inference_scaffold.py"),
            "--manifest-csv",
            str(manifest_path),
            "--labels-csv",
            str(labels_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_vindr_inference_scaffold_writes_summary_without_predictions(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_inputs(tmp_path)

    result = _run_scaffold(tmp_path, manifest_path, labels_path)

    assert result.returncode == 0
    summary_path = tmp_path / "out" / "vindr_inference_scaffold_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "medshiftlab.vindr_inference_scaffold.v1"
    assert summary["claim_level"]["model_loaded"] is False
    assert summary["claim_level"]["inference_completed"] is False
    assert summary["claim_level"]["metrics_completed"] is False
    assert summary["outputs"]["predictions_csv"] is None
    assert summary["input_summary"]["manifest"]["n_images_found"] == 2
    assert summary["input_summary"]["labels"]["positive_counts"]["Atelectasis"] == 1


def test_vindr_inference_scaffold_fails_when_manifest_has_missing_images(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_inputs(tmp_path)
    manifest = pd.read_csv(manifest_path)
    manifest.loc[1, "image_found"] = False
    manifest.to_csv(manifest_path, index=False)

    result = _run_scaffold(tmp_path, manifest_path, labels_path)

    assert result.returncode != 0
    assert "manifest contains missing images" in result.stdout
    assert not (tmp_path / "out" / "vindr_inference_scaffold_summary.json").exists()


def test_vindr_inference_scaffold_fails_when_manifest_uses_absolute_paths(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_inputs(tmp_path)
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "image_path"] = "/private/img001.jpg"
    manifest.to_csv(manifest_path, index=False)

    result = _run_scaffold(tmp_path, manifest_path, labels_path)

    assert result.returncode != 0
    assert "must be relative" in result.stdout


def test_vindr_inference_scaffold_fails_when_labels_do_not_match_manifest(tmp_path: Path) -> None:
    manifest_path, labels_path = _write_inputs(tmp_path)
    labels = pd.read_csv(labels_path)
    labels = labels[labels["sample_id"] != "vindr_img002"]
    labels.to_csv(labels_path, index=False)

    result = _run_scaffold(tmp_path, manifest_path, labels_path)

    assert result.returncode != 0
    assert "labels are missing manifest sample rows" in result.stdout
