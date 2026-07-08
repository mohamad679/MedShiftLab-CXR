from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_bootstrap_uncertainty_cli_writes_summary_outputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]

    predictions = {
        "schema_version": "medshiftlab.prediction.v1",
        "model_name": "synthetic-model",
        "model_version": "synthetic:v1",
        "preprocessing_version": "synthetic-preprocessing",
        "preprocessing_config": {
            "image_preprocessing": {
                "normalization": "torchxrayvision",
            }
        },
        "label_names": ["Finding A", "Finding B"],
        "records": [
            {
                "sample_id": "s1",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.90, 0.10],
            },
            {
                "sample_id": "s2",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.80, 0.20],
            },
            {
                "sample_id": "s3",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.20, 0.85],
            },
            {
                "sample_id": "s4",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.10, 0.75],
            },
            {
                "sample_id": "s5",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.70, 0.30],
            },
            {
                "sample_id": "s6",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.30, 0.70],
            },
        ],
    }

    labels = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "Finding A": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "Finding B": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )

    split = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": [
                "calibration",
                "calibration",
                "calibration",
                "evaluation",
                "evaluation",
                "evaluation",
            ],
        }
    )

    calibrated_eval = {
        "schema_version": "medshiftlab.calibrated_threshold_eval.v1",
        "dataset_name": "synthetic",
        "calibration_best_thresholds": [
            {"label": "Finding A", "threshold": 0.5},
            {"label": "Finding B", "threshold": 0.5},
        ],
    }

    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.csv"
    split_path = tmp_path / "split.csv"
    calibrated_eval_path = tmp_path / "calibrated_eval.json"
    output_json = tmp_path / "bootstrap_uncertainty.json"
    output_csv = tmp_path / "bootstrap_uncertainty.csv"

    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    labels.to_csv(labels_path, index=False)
    split.to_csv(split_path, index=False)
    calibrated_eval_path.write_text(json.dumps(calibrated_eval), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_bootstrap_uncertainty.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--split-csv",
            str(split_path),
            "--calibrated-threshold-eval",
            str(calibrated_eval_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--n-bootstrap",
            "25",
            "--bootstrap-seed",
            "123",
            "--notes",
            "synthetic bootstrap uncertainty test",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "medshiftlab.bootstrap_uncertainty.v1" in result.stdout
    assert output_json.exists()
    assert output_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "medshiftlab.bootstrap_uncertainty.v1"
    assert payload["dataset_name"] == "synthetic"
    assert payload["n_evaluation_records"] == 3
    assert payload["n_bootstrap"] == 25

    summary = pd.read_csv(output_csv)
    assert len(summary) == 8
    assert set(summary["threshold_source"]) == {"calibration_selected", "default_0.5"}
    assert set(summary["metric"]) == {
        "mean_f1",
        "mean_sensitivity",
        "mean_specificity",
        "mean_balanced_accuracy",
    }
    assert all(summary["ci_lower_2_5"] <= summary["ci_upper_97_5"])
