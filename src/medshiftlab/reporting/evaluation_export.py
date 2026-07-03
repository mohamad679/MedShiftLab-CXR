"""File export utilities for completed MedShiftLab-CXR evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from medshiftlab.evaluation.report import EvaluationReport


_LABEL_METRICS_HEADER = (
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
)


def write_evaluation_report_json(
    report: EvaluationReport, output_path: str | Path
) -> Path:
    """Write a complete evaluation report as pretty-printed JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_evaluation_label_metrics_csv(
    report: EvaluationReport, output_path: str | Path
) -> Path:
    """Write one stable CSV row per label in an evaluation report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_LABEL_METRICS_HEADER)
        writer.writeheader()
        writer.writerows(report.to_flat_rows())
    return path


def write_evaluation_report_bundle(
    report: EvaluationReport,
    output_dir: str | Path,
    *,
    json_filename: str = "evaluation_report.json",
    csv_filename: str = "evaluation_label_metrics.csv",
) -> dict[str, Path]:
    """Write JSON and label-metrics CSV files into one output directory."""

    directory = Path(output_dir)
    return {
        "json": write_evaluation_report_json(report, directory / json_filename),
        "csv": write_evaluation_label_metrics_csv(report, directory / csv_filename),
    }
