"""Synthetic-only tests for Phase 10 robustness analysis."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from medshiftlab.evaluation import (
    analyze_subgroups,
    bootstrap_label_metric_intervals,
    create_evaluation_report,
    summarize_failure_cases,
    summarize_label_calibration,
)
from medshiftlab.models import PredictionBatch, PredictionRecord
from medshiftlab.reporting import write_prediction_batch_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_bins_are_correct_and_deterministic() -> None:
    first = summarize_label_calibration(
        "Finding",
        [0, 0, 1, 1],
        [0.0, 0.2, 0.5, 1.0],
        n_bins=2,
    )
    second = summarize_label_calibration(
        "Finding",
        [0, 0, 1, 1],
        [0.0, 0.2, 0.5, 1.0],
        n_bins=2,
    )

    assert first == second
    assert [item.count for item in first.bins] == [2, 2]
    assert first.bins[0].mean_score == pytest.approx(0.1)
    assert first.bins[0].mean_target == 0.0
    assert first.bins[1].includes_upper_bound is True
    assert first.ece == pytest.approx(0.175)
    assert first.brier_score == pytest.approx(0.0725)


def test_calibration_retains_empty_bins() -> None:
    summary = summarize_label_calibration("Finding", [0, 1], [0.1, 0.9], n_bins=4)
    assert [item.count for item in summary.bins] == [1, 0, 0, 1]
    assert summary.bins[1].mean_score is None


def test_bootstrap_ci_is_deterministic() -> None:
    kwargs = {
        "label_name": "Finding",
        "y_true": [0, 1, 0, 1, 0, 1],
        "y_score": [0.1, 0.8, 0.2, 0.7, 0.4, 0.9],
        "metrics": ("auroc", "brier_score", "ece"),
        "iterations": 40,
        "seed": 17,
        "n_bins": 3,
    }
    first = bootstrap_label_metric_intervals(**kwargs)
    second = bootstrap_label_metric_intervals(**kwargs)
    assert first == second
    assert all(item.iterations == 40 for item in first)
    assert all(item.resampling_unit == "sample" for item in first)


def test_bootstrap_uses_patient_clusters_when_complete_ids_exist() -> None:
    intervals = bootstrap_label_metric_intervals(
        "Finding",
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        metrics=("brier_score",),
        iterations=10,
        seed=3,
        patient_ids=["p1", "p1", "p2", "p2"],
    )
    assert intervals[0].resampling_unit == "patient"


def test_unsupported_bootstrap_metric_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported bootstrap metric.*accuracy"):
        bootstrap_label_metric_intervals(
            "Finding",
            [0, 1],
            [0.1, 0.9],
            metrics=("accuracy",),
            iterations=10,
            seed=1,
        )


def _subgroup_rows() -> list[dict[str, object]]:
    return [
        {"sex": "F", "true_Finding": 0, "score_Finding": 0.1},
        {"sex": "F", "true_Finding": 1, "score_Finding": 0.9},
        {"sex": "M", "true_Finding": 1, "score_Finding": 0.8},
    ]


def test_subgroup_grouping_and_small_group_skip_reasons() -> None:
    report = analyze_subgroups(
        _subgroup_rows(),
        labels=("Finding",),
        subgroup_columns=("sex",),
        minimum_subgroup_size=2,
    )
    assert len(report.results) == 1
    assert report.results[0].subgroup_value == "F"
    assert report.results[0].metrics.auroc == 1.0
    assert len(report.skipped) == 1
    assert report.skipped[0].subgroup_value == "M"
    assert report.skipped[0].reason == "insufficient_available_samples"


def test_missing_subgroup_columns_fail_clearly() -> None:
    with pytest.raises(ValueError, match="Missing requested subgroup column.*age_group"):
        analyze_subgroups(
            _subgroup_rows(),
            labels=("Finding",),
            subgroup_columns=("age_group",),
            minimum_subgroup_size=1,
        )


def test_private_absolute_subgroup_values_are_rejected() -> None:
    rows = [{"site": "/synthetic/private/data", "true_Finding": 0, "score_Finding": 0.1}]
    with pytest.raises(ValueError, match="Absolute paths"):
        analyze_subgroups(
            rows,
            labels=("Finding",),
            subgroup_columns=("site",),
            minimum_subgroup_size=1,
        )


def test_failure_case_summary_is_schema_valid() -> None:
    current = create_evaluation_report(
        dataset_name="synthetic",
        model_name="mock",
        y_true_by_label={"Finding": [0, 1, 0, 1]},
        y_score_by_label={"Finding": [0.8, 0.2, 0.7, 0.3]},
        n_bins=2,
    )
    baseline = create_evaluation_report(
        dataset_name="synthetic",
        model_name="baseline",
        y_true_by_label={"Finding": [0, 1, 0, 1]},
        y_score_by_label={"Finding": [0.1, 0.9, 0.2, 0.8]},
        n_bins=2,
    )
    subgroup_report = analyze_subgroups(
        _subgroup_rows(),
        labels=("Finding",),
        subgroup_columns=("sex",),
        minimum_subgroup_size=2,
    )
    summary = summarize_failure_cases(
        current,
        subgroup_report=subgroup_report,
        baseline_report=baseline,
        degradation_metric="auroc",
        degradation_threshold=0.1,
        maximum_ece=0.1,
    )
    validated = type(summary).model_validate(summary.model_dump())
    assert validated.clinical_interpretation_performed is False
    assert validated.metric_degradations[0].label_name == "Finding"
    assert validated.poor_calibration
    assert validated.coverage_issues[0].issue_type == "subgroup_label_skipped"


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    records = [
        PredictionRecord(
            sample_id=f"s{index}",
            model_name="mock",
            dataset_name="synthetic",
            label_names=("Finding",),
            probabilities=(score,),
            patient_id=f"p{index // 2}",
            image_path=f"/synthetic/private/images/s{index}.png",
        )
        for index, score in enumerate((0.1, 0.8, 0.2, 0.9, 0.3, 0.7))
    ]
    batch = PredictionBatch(
        model_name="mock",
        model_version="v1",
        adapter_name="mock",
        preprocessing_version="v1",
        preprocessing_config={"mode": "synthetic"},
        records=records,
        label_names=("Finding",),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    predictions = write_prediction_batch_json(batch, tmp_path / "predictions.json")
    labels = tmp_path / "labels.csv"
    metadata = tmp_path / "metadata.csv"
    labels.write_text(
        "sample_id,Finding\ns0,0\ns1,1\ns2,0\ns3,1\ns4,0\ns5,1\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "sample_id,sex,age_group\ns0,F,<40\ns1,F,<40\ns2,M,40-64\n"
        "s3,M,40-64\ns4,,>=65\ns5,,>=65\n",
        encoding="utf-8",
    )
    return predictions, labels, metadata


def test_script_runs_on_synthetic_files_without_path_leakage(tmp_path: Path) -> None:
    predictions, labels, metadata = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "local_robustness"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_robustness_calibration_analysis.py",
            "--predictions", str(predictions),
            "--labels-csv", str(labels),
            "--metadata-csv", str(metadata),
            "--output-dir", str(output_dir),
            "--export-calibration-csv",
            "--bootstrap-iters", "20",
            "--seed", "5",
            "--subgroup-columns", "sex", "age_group", "dataset_name",
            "--minimum-subgroup-size", "2",
            "--limit", "6",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["manual_local_only"] is True
    assert summary["real_inference_performed"] is False
    assert summary["bootstrap_intervals"] == 3
    assert summary["support_status"]["bootstrap_resampling"] == "patient"
    assert set(summary["outputs_written"]) == {
        "json", "subgroups_csv", "bootstrap_csv", "calibration_csv"
    }
    output_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output_dir.iterdir()
    )
    assert "/synthetic/private" not in result.stdout
    assert "/synthetic/private" not in output_text


def test_script_rejects_unsupported_bootstrap_metric(tmp_path: Path) -> None:
    predictions, labels, _ = _write_synthetic_inputs(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_robustness_calibration_analysis.py",
            "--predictions", str(predictions),
            "--labels-csv", str(labels),
            "--bootstrap-iters", "5",
            "--bootstrap-metrics", "accuracy",
            "--subgroup-columns", "dataset_name",
            "--minimum-subgroup-size", "1",
            "--limit", "6",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Unsupported bootstrap metric" in result.stderr
