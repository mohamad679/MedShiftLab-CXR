"""Safe aggregate exports for Phase 10 robustness reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from medshiftlab.evaluation.robustness import RobustnessAnalysisReport


def write_robustness_report_json(
    report: RobustnessAnalysisReport, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_calibration_bins_csv(
    report: RobustnessAnalysisReport, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dataset_name",
        "model_name",
        "label_name",
        "n_available",
        "n_bins",
        "ece",
        "brier_score",
        "bin_index",
        "lower_bound",
        "upper_bound",
        "includes_upper_bound",
        "count",
        "mean_score",
        "mean_target",
        "absolute_gap",
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for label_name, summary in report.calibration.items():
            for bin_data in summary.bins:
                writer.writerow(
                    {
                        "dataset_name": report.dataset_name,
                        "model_name": report.model_name,
                        "label_name": label_name,
                        "n_available": summary.n_available,
                        "n_bins": summary.n_bins,
                        "ece": summary.ece,
                        "brier_score": summary.brier_score,
                        **bin_data.model_dump(),
                    }
                )
    return path


def write_subgroup_metrics_csv(
    report: RobustnessAnalysisReport, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = tuple(
        key for key in report.subgroup_analysis.results[0].metrics.model_fields
    ) if report.subgroup_analysis.results else (
        "label_name", "n_available", "n_binary", "n_positive", "n_negative",
        "threshold", "auroc", "auprc", "brier_score", "ece", "f1",
        "sensitivity", "specificity", "true_positive", "false_positive",
        "true_negative", "false_negative",
    )
    fieldnames = ("subgroup_column", "subgroup_value", "group_size", *metric_fields)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in report.subgroup_analysis.results:
            writer.writerow(
                {
                    "subgroup_column": item.subgroup_column,
                    "subgroup_value": item.subgroup_value,
                    "group_size": item.group_size,
                    **item.metrics.model_dump(),
                }
            )
    return path


def write_bootstrap_intervals_csv(
    report: RobustnessAnalysisReport, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "label_name", "metric_name", "point_estimate", "lower", "upper",
        "confidence_level", "iterations", "valid_iterations", "seed",
        "resampling_unit",
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.model_dump() for item in report.bootstrap_intervals)
    return path


def write_robustness_report_bundle(
    report: RobustnessAnalysisReport,
    output_dir: str | Path,
    *,
    export_calibration_csv: bool = False,
) -> dict[str, Path]:
    directory = Path(output_dir)
    outputs = {
        "json": write_robustness_report_json(
            report, directory / "robustness_analysis.json"
        ),
        "subgroups_csv": write_subgroup_metrics_csv(
            report, directory / "subgroup_metrics.csv"
        ),
        "bootstrap_csv": write_bootstrap_intervals_csv(
            report, directory / "bootstrap_intervals.csv"
        ),
    }
    if export_calibration_csv:
        outputs["calibration_csv"] = write_calibration_bins_csv(
            report, directory / "calibration_bins.csv"
        )
    return outputs
