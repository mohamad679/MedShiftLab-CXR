"""Aggregate exports for cross-dataset bootstrap reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medshiftlab.experiments.cross_dataset_bootstrap import (
        CrossDatasetBootstrapReport,
    )


SUMMARY_JSON_FILENAME = "cross_dataset_bootstrap_summary.json"
SUMMARY_CSV_FILENAME = "cross_dataset_bootstrap_summary.csv"


def write_cross_dataset_bootstrap_json(
    report: "CrossDatasetBootstrapReport",
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_cross_dataset_bootstrap_csv(
    report: "CrossDatasetBootstrapReport",
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "reference_dataset_name",
        "external_dataset_name",
        "model_name",
        "reference_resampling_unit",
        "external_resampling_unit",
        "delta_definition",
        "label_name",
        "metric_name",
        "metric_direction",
        "reference_point_estimate",
        "reference_ci_lower",
        "reference_ci_upper",
        "reference_valid_iterations",
        "external_point_estimate",
        "external_ci_lower",
        "external_ci_upper",
        "external_valid_iterations",
        "delta_point_estimate",
        "delta_ci_lower",
        "delta_ci_upper",
        "delta_valid_iterations",
        "confidence_level",
        "iterations",
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in report.results:
            writer.writerow(
                {
                    "reference_dataset_name": report.reference_dataset_name,
                    "external_dataset_name": report.external_dataset_name,
                    "model_name": report.model_name,
                    "reference_resampling_unit": report.reference_resampling_unit,
                    "external_resampling_unit": report.external_resampling_unit,
                    "delta_definition": report.delta_definition,
                    **result.model_dump(),
                }
            )
    return path


def write_cross_dataset_bootstrap_bundle(
    report: "CrossDatasetBootstrapReport",
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir)
    return {
        "json": write_cross_dataset_bootstrap_json(
            report,
            directory / SUMMARY_JSON_FILENAME,
        ),
        "csv": write_cross_dataset_bootstrap_csv(
            report,
            directory / SUMMARY_CSV_FILENAME,
        ),
    }
