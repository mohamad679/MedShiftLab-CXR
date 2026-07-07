"""Focused tests for standardized real-prediction evaluation orchestration."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from medshiftlab.experiments import (
    PredictionEvaluationConfig,
    load_ground_truth_label_rows,
    run_prediction_batch_evaluation_from_files,
)
from medshiftlab.models import PREDICTION_SCHEMA_VERSION, PredictionBatch, PredictionRecord
from medshiftlab.reporting import (
    read_prediction_batch_json,
    write_prediction_batch_json,
    write_prediction_records_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    return environment


def _prediction_record(
    sample_id: str,
    *,
    atelectasis: float,
    cardiomegaly: float,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        dataset_name="chexpert",
        model_name="mock-cxr",
        image_path=f"images/{sample_id}.png",
        label_names=("Atelectasis", "Cardiomegaly"),
        probabilities=(atelectasis, cardiomegaly),
    )


def _prediction_batch(records: list[PredictionRecord]) -> PredictionBatch:
    return PredictionBatch(
        schema_version=PREDICTION_SCHEMA_VERSION,
        model_name="mock-cxr",
        model_version="mock-cxr:v1",
        adapter_name="mock-cxr-adapter",
        preprocessing_version="preprocess-v1",
        preprocessing_config={"output_mode": "grayscale"},
        records=records,
        label_names=("Atelectasis", "Cardiomegaly"),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_metadata={"manual_only": True},
    )


def _write_labels_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_id", "Atelectasis", "Cardiomegaly"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return labels_path


def test_prediction_json_evaluation_works(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch(
            [
                _prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9),
                _prediction_record("img002", atelectasis=0.9, cardiomegaly=0.1),
            ]
        ),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
            {"sample_id": "img002", "Atelectasis": 1, "Cardiomegaly": 0},
        ],
    )

    result, written_outputs = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_path,
        config=PredictionEvaluationConfig(limit=2, threshold=0.5, n_bins=2),
    )

    assert result.prediction_format == "json"
    assert result.report.metadata.dataset_name == "chexpert"
    assert result.report.metadata.model_name == "mock-cxr"
    assert result.report.label_metrics["Atelectasis"].auroc == 1.0
    assert result.report.label_metrics["Cardiomegaly"].auprc == 1.0
    assert result.accounting.evaluated_records == 2
    assert result.accounting.skipped_records == 0
    assert written_outputs == {}


def test_prediction_csv_evaluation_works(tmp_path: Path) -> None:
    prediction_batch = _prediction_batch(
        [
            _prediction_record("img001", atelectasis=0.2, cardiomegaly=0.8),
            _prediction_record("img002", atelectasis=0.8, cardiomegaly=0.2),
        ]
    )
    predictions_path = write_prediction_records_csv(
        prediction_batch,
        tmp_path / "predictions.csv",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
            {"sample_id": "img002", "Atelectasis": 1, "Cardiomegaly": 0},
        ],
    )
    output_json = tmp_path / "evaluation_report.json"
    output_csv = tmp_path / "evaluation_label_metrics.csv"

    result, written_outputs = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_path,
        config=PredictionEvaluationConfig(limit=2, threshold=0.5, n_bins=2),
        output_json=output_json,
        output_csv=output_csv,
    )

    assert result.prediction_format == "csv"
    assert written_outputs["json"] == output_json
    assert written_outputs["csv"] == output_csv
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["label_metrics"]["Atelectasis"]["true_positive"] == 1
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8", newline="")))
    assert rows[0]["true_positive"] == "1"


def test_sample_id_join_works_with_image_id_label_column(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "image_id,Atelectasis,Cardiomegaly\nimg001,0,1\n",
        encoding="utf-8",
    )

    result, _ = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_path,
        config=PredictionEvaluationConfig(limit=1),
    )

    assert result.accounting.evaluated_records == 1


def test_missing_label_rows_fail_clearly(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [{"sample_id": "img999", "Atelectasis": 0, "Cardiomegaly": 1}],
    )

    with pytest.raises(ValueError, match="missing_labels=1, missing_predictions=1"):
        run_prediction_batch_evaluation_from_files(
            predictions_path=predictions_path,
            labels_csv_path=labels_path,
            config=PredictionEvaluationConfig(limit=1),
        )


def test_duplicate_label_rows_fail_clearly(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
        ],
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_prediction_batch_evaluation_from_files(
            predictions_path=predictions_path,
            labels_csv_path=labels_path,
            config=PredictionEvaluationConfig(limit=1),
        )


def test_label_name_mismatch_fails_clearly(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "sample_id,Atelectasis\nimg001,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required label column"):
        run_prediction_batch_evaluation_from_files(
            predictions_path=predictions_path,
            labels_csv_path=labels_path,
            config=PredictionEvaluationConfig(limit=1),
        )


def test_metric_outputs_are_schema_valid_and_threshold_is_applied(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch(
            [
                _prediction_record("img001", atelectasis=0.4, cardiomegaly=0.6),
                _prediction_record("img002", atelectasis=0.6, cardiomegaly=0.4),
            ]
        ),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
            {"sample_id": "img002", "Atelectasis": 1, "Cardiomegaly": 0},
        ],
    )

    result, _ = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_path,
        config=PredictionEvaluationConfig(limit=2, threshold=0.55),
    )

    metrics = result.report.label_metrics["Atelectasis"]
    assert metrics.threshold == 0.55
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0


def test_no_private_paths_appear_in_exported_reports(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [{"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1}],
    )
    output_json = tmp_path / "evaluation_report.json"
    output_csv = tmp_path / "evaluation_label_metrics.csv"

    _, _written_outputs = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_path,
        config=PredictionEvaluationConfig(limit=1),
        output_json=output_json,
        output_csv=output_csv,
    )

    json_text = output_json.read_text(encoding="utf-8")
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "/Users/mohsenshamsijazeb" not in json_text
    assert "/Users/mohsenshamsijazeb" not in csv_text
    assert "images/img001.png" not in json_text


def test_load_ground_truth_label_rows_requires_sample_identifier(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "Atelectasis,Cardiomegaly\n0,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_id"):
        load_ground_truth_label_rows(
            labels_path,
            label_names=("Atelectasis", "Cardiomegaly"),
        )


def test_script_works_without_real_datasets(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch(
            [
                _prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9),
                _prediction_record("img002", atelectasis=0.9, cardiomegaly=0.1),
            ]
        ),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [
            {"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1},
            {"sample_id": "img002", "Atelectasis": 1, "Cardiomegaly": 0},
        ],
    )
    output_json = tmp_path / "evaluation_report.json"
    output_csv = tmp_path / "evaluation_label_metrics.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_prediction_evaluation.py",
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--limit",
            "2",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["manual_only"] is True
    assert summary["evaluated_records"] == 2
    assert summary["prediction_format"] == "json"
    assert summary["support_status"]["bootstrap_ci"] == "not_implemented"
    assert output_json.is_file()
    assert output_csv.is_file()


def test_script_reports_bootstrap_not_implemented(tmp_path: Path) -> None:
    predictions_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    labels_path = _write_labels_csv(
        tmp_path,
        [{"sample_id": "img001", "Atelectasis": 0, "Cardiomegaly": 1}],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_prediction_evaluation.py",
            "--predictions",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--bootstrap-iters",
            "10",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not implemented" in result.stderr.lower()


def test_prediction_json_roundtrip_still_validates(tmp_path: Path) -> None:
    original_path = write_prediction_batch_json(
        _prediction_batch([_prediction_record("img001", atelectasis=0.1, cardiomegaly=0.9)]),
        tmp_path / "predictions.json",
    )
    reloaded = read_prediction_batch_json(original_path)

    assert reloaded.schema_version == PREDICTION_SCHEMA_VERSION
    assert reloaded.records[0].sample_id == "img001"
