"""Tests for MedShiftLab-CXR evaluation report file exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from medshiftlab.evaluation import (
    EvaluationAggregateMetrics,
    EvaluationReport,
    EvaluationRunMetadata,
    create_evaluation_report,
)
from medshiftlab.reporting import (
    write_evaluation_label_metrics_csv,
    write_evaluation_report_bundle,
    write_evaluation_report_json,
)


def _report() -> EvaluationReport:
    return create_evaluation_report(
        dataset_name="CheXpert",
        model_name="mock-pretrained-cxr",
        split="validation",
        uncertainty_strategy="U-ignore",
        threshold=0.5,
        n_bins=2,
        y_true_by_label={
            "Atelectasis": [0, 1, 0, 1],
            "Cardiomegaly": [0, 1, 1, 0],
        },
        y_score_by_label={
            "Atelectasis": [0.1, 0.9, 0.2, 0.8],
            "Cardiomegaly": [0.2, 0.8, 0.7, 0.3],
        },
    )


def test_json_export_writes_complete_report_metadata(tmp_path: Path) -> None:
    output_path = write_evaluation_report_json(
        _report(), tmp_path / "evaluation.json"
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["dataset_name"] == "CheXpert"
    assert payload["metadata"]["model_name"] == "mock-pretrained-cxr"
    assert set(payload["label_metrics"]) == {"Atelectasis", "Cardiomegaly"}


def test_csv_export_writes_one_row_per_label(tmp_path: Path) -> None:
    output_path = write_evaluation_label_metrics_csv(
        _report(), tmp_path / "metrics.csv"
    )

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert len(rows) == 2
    assert [row["label_name"] for row in rows] == [
        "Atelectasis",
        "Cardiomegaly",
    ]
    assert rows[0]["dataset_name"] == "CheXpert"
    assert rows[0]["model_name"] == "mock-pretrained-cxr"
    assert reader.fieldnames == [
        "dataset_name",
        "model_name",
        "split",
        "uncertainty_strategy",
        "threshold",
        "n_bins",
        "label_name",
        "n_available",
        "n_binary",
        "n_positive",
        "n_negative",
        "auroc",
        "auprc",
        "brier_score",
        "ece",
        "f1",
        "sensitivity",
        "specificity",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]


def test_csv_export_writes_header_for_report_without_labels(tmp_path: Path) -> None:
    report = EvaluationReport(
        metadata=EvaluationRunMetadata(
            dataset_name="CheXpert",
            model_name="mock-pretrained-cxr",
            threshold=0.5,
            n_bins=10,
        ),
        aggregate_metrics=EvaluationAggregateMetrics(n_labels=0),
        label_metrics={},
    )

    output_path = write_evaluation_label_metrics_csv(
        report, tmp_path / "empty-metrics.csv"
    )

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames is not None
        assert "label_name" in reader.fieldnames
        assert list(reader) == []


def test_bundle_export_creates_both_default_files(tmp_path: Path) -> None:
    paths = write_evaluation_report_bundle(_report(), tmp_path / "bundle")

    assert paths["json"] == tmp_path / "bundle" / "evaluation_report.json"
    assert paths["csv"] == tmp_path / "bundle" / "evaluation_label_metrics.csv"
    assert paths["json"].is_file()
    assert paths["csv"].is_file()


def test_exporters_create_parent_directories(tmp_path: Path) -> None:
    json_path = write_evaluation_report_json(
        _report(), tmp_path / "nested" / "json" / "report.json"
    )
    csv_path = write_evaluation_label_metrics_csv(
        _report(), tmp_path / "nested" / "csv" / "metrics.csv"
    )

    assert json_path.is_file()
    assert csv_path.is_file()


def test_exporters_return_path_objects(tmp_path: Path) -> None:
    json_path = write_evaluation_report_json(_report(), str(tmp_path / "report.json"))
    csv_path = write_evaluation_label_metrics_csv(_report(), str(tmp_path / "metrics.csv"))
    bundle_paths = write_evaluation_report_bundle(_report(), str(tmp_path / "bundle"))

    assert isinstance(json_path, Path)
    assert isinstance(csv_path, Path)
    assert all(isinstance(path, Path) for path in bundle_paths.values())


def test_bundle_export_supports_custom_filenames(tmp_path: Path) -> None:
    paths = write_evaluation_report_bundle(
        _report(),
        tmp_path,
        json_filename="custom-report.json",
        csv_filename="custom-metrics.csv",
    )

    assert paths == {
        "json": tmp_path / "custom-report.json",
        "csv": tmp_path / "custom-metrics.csv",
    }
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
