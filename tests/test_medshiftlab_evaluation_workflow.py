from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    predictions = {
        "schema_version": "medshiftlab.prediction.v1",
        "model_name": "synthetic-model",
        "model_version": "synthetic:v1",
        "preprocessing_version": "synthetic-preprocessing",
        "preprocessing_config": {"image_preprocessing": {"normalization": "torchxrayvision"}},
        "label_names": ["Finding A", "Finding B"],
        "records": [
            {"sample_id": "s1", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.95, 0.05]},
            {"sample_id": "s2", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.85, 0.15]},
            {"sample_id": "s3", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.20, 0.90]},
            {"sample_id": "s4", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.10, 0.80]},
            {"sample_id": "s5", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.88, 0.12]},
            {"sample_id": "s6", "dataset_name": "synthetic", "label_names": ["Finding A", "Finding B"], "probabilities": [0.12, 0.88]},
        ],
    }
    labels = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "Finding A": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "Finding B": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "view_position": ["AP", "AP", "PA", "AP", "PA", "AP"],
            "age_bucket": ["40-59", "40-59", "60-79", "60-79", "<40", "<40"],
        }
    )
    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.csv"
    metadata_path = tmp_path / "metadata.csv"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    labels.to_csv(labels_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return predictions_path, labels_path, metadata_path


def test_evaluation_workflow_dry_run_prints_all_commands(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    predictions_path, labels_path, metadata_path = _write_synthetic_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_evaluation_workflow.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--metadata-csv",
            str(metadata_path),
            "--subgroup-columns",
            "sex",
            "view_position",
            "age_bucket",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--n-bootstrap",
            "10",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "medshiftlab.evaluation_workflow.v1"
    assert payload["execute"] is False
    assert len(payload["commands"]) == 4
    assert any("run_threshold_sweep.py" in command for command in payload["commands"])
    assert any("run_calibrated_threshold_evaluation.py" in command for command in payload["commands"])
    assert any("run_bootstrap_uncertainty.py" in command for command in payload["commands"])
    assert any("run_subgroup_audit.py" in command for command in payload["commands"])


def test_evaluation_workflow_execute_writes_expected_outputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    predictions_path, labels_path, metadata_path = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_evaluation_workflow.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--metadata-csv",
            str(metadata_path),
            "--subgroup-columns",
            "sex",
            "view_position",
            "age_bucket",
            "--output-dir",
            str(output_dir),
            "--threshold-step",
            "0.5",
            "--n-bootstrap",
            "10",
            "--bootstrap-seed",
            "123",
            "--execute",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    expected_outputs = [
        "threshold_sweep.json",
        "threshold_sweep.csv",
        "threshold_sweep_best.csv",
        "calibrated_threshold_eval.json",
        "calibration_best.csv",
        "evaluation.csv",
        "split.csv",
        "bootstrap_uncertainty.json",
        "bootstrap_uncertainty.csv",
        "subgroup_audit.json",
        "subgroup_label_metrics.csv",
        "subgroup_aggregate.csv",
    ]
    for filename in expected_outputs:
        assert (output_dir / filename).exists(), filename
