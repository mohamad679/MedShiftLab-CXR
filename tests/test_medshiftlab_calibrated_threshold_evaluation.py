from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_calibrated_threshold_evaluation_cli_writes_outputs(tmp_path: Path) -> None:
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
                "probabilities": [0.95, 0.05],
            },
            {
                "sample_id": "s2",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.85, 0.15],
            },
            {
                "sample_id": "s3",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.20, 0.90],
            },
            {
                "sample_id": "s4",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.10, 0.80],
            },
            {
                "sample_id": "s5",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.88, 0.12],
            },
            {
                "sample_id": "s6",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.12, 0.88],
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

    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.csv"
    output_json = tmp_path / "calibrated_threshold_eval.json"
    output_calibration_best_csv = tmp_path / "calibration_best.csv"
    output_evaluation_csv = tmp_path / "evaluation.csv"
    output_split_csv = tmp_path / "split.csv"

    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    labels.to_csv(labels_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_calibrated_threshold_evaluation.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--output-json",
            str(output_json),
            "--output-calibration-best-csv",
            str(output_calibration_best_csv),
            "--output-evaluation-csv",
            str(output_evaluation_csv),
            "--output-split-csv",
            str(output_split_csv),
            "--calibration-fraction",
            "0.5",
            "--threshold-start",
            "0.0",
            "--threshold-stop",
            "1.0",
            "--threshold-step",
            "0.01",
            "--notes",
            "synthetic calibrated threshold evaluation test",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "medshiftlab.calibrated_threshold_eval.v1" in result.stdout
    assert output_json.exists()
    assert output_calibration_best_csv.exists()
    assert output_evaluation_csv.exists()
    assert output_split_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "medshiftlab.calibrated_threshold_eval.v1"
    assert payload["dataset_name"] == "synthetic"
    assert payload["n_records"] == 6
    assert payload["normalization"] == "torchxrayvision"
    assert payload["split"]["calibration_records"] == 3
    assert payload["split"]["evaluation_records"] == 3

    split = pd.read_csv(output_split_csv)
    assert set(split["split"]) == {"calibration", "evaluation"}

    calibration_best = pd.read_csv(output_calibration_best_csv)
    evaluation = pd.read_csv(output_evaluation_csv)

    assert set(calibration_best["label"]) == {"Finding A", "Finding B"}
    assert set(evaluation["threshold_source"]) == {"calibration_selected", "default_0.5"}
