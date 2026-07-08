from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_threshold_sweep_cli_writes_outputs(tmp_path: Path) -> None:
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
                "probabilities": [0.40, 0.70],
            },
            {
                "sample_id": "s4",
                "dataset_name": "synthetic",
                "label_names": ["Finding A", "Finding B"],
                "probabilities": [0.10, 0.60],
            },
        ],
    }

    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.csv"
    output_json = tmp_path / "threshold_sweep.json"
    output_csv = tmp_path / "threshold_sweep.csv"
    output_best_csv = tmp_path / "threshold_sweep_best.csv"

    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "Finding A": [1.0, 1.0, 0.0, 0.0],
            "Finding B": [0.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(labels_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_threshold_sweep.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-best-csv",
            str(output_best_csv),
            "--threshold-start",
            "0.0",
            "--threshold-stop",
            "1.0",
            "--threshold-step",
            "0.01",
            "--notes",
            "synthetic threshold sweep test",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "medshiftlab.threshold_sweep.v1" in result.stdout
    assert output_json.exists()
    assert output_csv.exists()
    assert output_best_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "medshiftlab.threshold_sweep.v1"
    assert payload["dataset_name"] == "synthetic"
    assert payload["n_records"] == 4
    assert payload["normalization"] == "torchxrayvision"

    best = pd.read_csv(output_best_csv)
    assert set(best["label"]) == {"Finding A", "Finding B"}
    assert all(best["f1"] == 1.0)
    assert all(best["balanced_accuracy"] == 1.0)
