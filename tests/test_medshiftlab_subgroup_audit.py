from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_subgroup_audit_cli_writes_label_and_aggregate_outputs(tmp_path: Path) -> None:
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

    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "view_position": ["AP", "AP", "PA", "AP", "PA", "AP"],
            "age_bucket": ["40-59", "40-59", "60-79", "60-79", "<40", "<40"],
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
    metadata_path = tmp_path / "metadata.csv"
    calibrated_eval_path = tmp_path / "calibrated_eval.json"
    output_json = tmp_path / "subgroup_audit.json"
    output_label_metrics_csv = tmp_path / "subgroup_label_metrics.csv"
    output_aggregate_csv = tmp_path / "subgroup_aggregate.csv"

    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    labels.to_csv(labels_path, index=False)
    split.to_csv(split_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    calibrated_eval_path.write_text(json.dumps(calibrated_eval), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_subgroup_audit.py"),
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--split-csv",
            str(split_path),
            "--calibrated-threshold-eval",
            str(calibrated_eval_path),
            "--metadata-csv",
            str(metadata_path),
            "--subgroup-columns",
            "sex",
            "view_position",
            "age_bucket",
            "--output-json",
            str(output_json),
            "--output-label-metrics-csv",
            str(output_label_metrics_csv),
            "--output-aggregate-csv",
            str(output_aggregate_csv),
            "--notes",
            "synthetic subgroup audit test",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "medshiftlab.subgroup_audit.v1" in result.stdout
    assert output_json.exists()
    assert output_label_metrics_csv.exists()
    assert output_aggregate_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "medshiftlab.subgroup_audit.v1"
    assert payload["dataset_name"] == "synthetic"
    assert payload["n_split_records"] == 3
    assert payload["subgroup_variables"] == ["sex", "view_position", "age_bucket"]

    label_metrics = pd.read_csv(output_label_metrics_csv)
    aggregate = pd.read_csv(output_aggregate_csv)

    assert set(label_metrics["threshold_source"]) == {"calibration_selected", "default_0.5"}
    assert set(aggregate["threshold_source"]) == {"calibration_selected", "default_0.5"}
    assert {"sex", "view_position", "age_bucket"}.issubset(set(aggregate["subgroup_var"]))
