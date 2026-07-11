from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from medshiftlab.experiments import run_cross_dataset_bootstrap_from_files
from medshiftlab.models import PREDICTION_SCHEMA_VERSION, PredictionBatch, PredictionRecord
from medshiftlab.reporting import (
    write_cross_dataset_bootstrap_bundle,
    write_prediction_batch_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Finding A", "Finding B")
TARGETS_A = (1, 1, 1, 0, 0, 0, 1, 0)
TARGETS_B = (0, 0, 0, 1, 1, 1, 0, 1)
REFERENCE_A = (0.95, 0.90, 0.82, 0.10, 0.20, 0.28, 0.78, 0.16)
REFERENCE_B = (0.05, 0.08, 0.18, 0.92, 0.80, 0.72, 0.22, 0.84)
EXTERNAL_A = (0.45, 0.40, 0.35, 0.72, 0.68, 0.60, 0.30, 0.55)
EXTERNAL_B = (0.55, 0.60, 0.65, 0.28, 0.32, 0.40, 0.70, 0.45)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    return environment


def _write_dataset(
    tmp_path: Path,
    *,
    stem: str,
    dataset_name: str,
    scores_a: tuple[float, ...] = REFERENCE_A,
    scores_b: tuple[float, ...] = REFERENCE_B,
    targets_a: tuple[int, ...] = TARGETS_A,
    targets_b: tuple[int, ...] = TARGETS_B,
    labels: tuple[str, ...] = LABELS,
    patient_ids: tuple[str | None, ...] | None = None,
    model_name: str = "mock-cxr",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if patient_ids is None:
        patient_ids = tuple(f"patient-{stem}-{index // 2}" for index in range(len(scores_a)))

    records = []
    for index, score_a in enumerate(scores_a):
        probabilities = (score_a, scores_b[index]) if labels == LABELS else (score_a,)
        records.append(
            PredictionRecord(
                sample_id=f"{stem}_sample_{index}",
                patient_id=patient_ids[index],
                dataset_name=dataset_name,
                model_name=model_name,
                image_path=str(tmp_path / f"{stem}_image_{index}.png"),
                label_names=labels,
                probabilities=probabilities,
            )
        )

    batch = PredictionBatch(
        schema_version=PREDICTION_SCHEMA_VERSION,
        model_name=model_name,
        model_version="mock-cxr:v1",
        adapter_name="mock-adapter",
        preprocessing_version="preprocess-v1",
        preprocessing_config={"normalization": "synthetic"},
        records=records,
        label_names=labels,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        run_metadata={"manual_only": True},
    )
    predictions_path = write_prediction_batch_json(
        batch,
        tmp_path / f"{stem}_predictions.json",
    )

    labels_path = tmp_path / f"{stem}_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("sample_id", *labels))
        writer.writeheader()
        for index, target_a in enumerate(targets_a):
            row: dict[str, object] = {
                "sample_id": f"{stem}_sample_{index}",
                labels[0]: target_a,
            }
            if labels == LABELS:
                row[labels[1]] = targets_b[index]
            writer.writerow(row)
    return predictions_path, labels_path


def _run_report(
    tmp_path: Path,
    *,
    iterations: int = 80,
    seed: int = 2026,
    reference_patient_ids: tuple[str | None, ...] | None = None,
    external_patient_ids: tuple[str | None, ...] | None = None,
    metrics: tuple[str, ...] = ("auroc", "auprc", "brier_score", "ece"),
):
    reference_predictions, reference_labels = _write_dataset(
        tmp_path,
        stem="reference",
        dataset_name="reference-dataset",
        scores_a=REFERENCE_A,
        scores_b=REFERENCE_B,
        patient_ids=reference_patient_ids,
    )
    external_predictions, external_labels = _write_dataset(
        tmp_path,
        stem="external",
        dataset_name="external-dataset",
        scores_a=EXTERNAL_A,
        scores_b=EXTERNAL_B,
        patient_ids=external_patient_ids,
    )
    return run_cross_dataset_bootstrap_from_files(
        reference_predictions_path=reference_predictions,
        reference_labels_csv_path=reference_labels,
        external_predictions_path=external_predictions,
        external_labels_csv_path=external_labels,
        iterations=iterations,
        seed=seed,
        confidence_level=0.95,
        n_bins=4,
        metrics=metrics,
    )


def test_cross_dataset_bootstrap_core_behavior(tmp_path: Path) -> None:
    external_patient_ids = tuple(
        None if index == 0 else f"patient-external-{index // 2}"
        for index in range(len(TARGETS_A))
    )
    report = _run_report(
        tmp_path,
        external_patient_ids=external_patient_ids,
        iterations=80,
    )

    assert report.schema_version == "medshiftlab.cross_dataset_bootstrap.v1"
    assert report.reference_resampling_unit == "patient"
    assert report.external_resampling_unit == "sample"
    assert report.delta_definition == "external_minus_reference"
    assert report.independent_dataset_bootstrap is True
    assert report.real_inference_performed is False
    assert report.manual_local_only is True

    directions = {
        result.metric_name: result.metric_direction
        for result in report.results
        if result.label_name == "Finding A"
    }
    assert directions == {
        "auroc": "higher_is_better",
        "auprc": "higher_is_better",
        "brier_score": "lower_is_better",
        "ece": "lower_is_better",
    }

    for result in report.results:
        assert result.reference_valid_iterations <= report.iterations
        assert result.external_valid_iterations <= report.iterations
        assert result.delta_valid_iterations <= report.iterations
        if result.reference_ci_lower is not None and result.reference_ci_upper is not None:
            assert result.reference_ci_lower <= result.reference_ci_upper
        if result.external_ci_lower is not None and result.external_ci_upper is not None:
            assert result.external_ci_lower <= result.external_ci_upper
        if result.delta_ci_lower is not None and result.delta_ci_upper is not None:
            assert result.delta_ci_lower <= result.delta_ci_upper
        if (
            result.reference_point_estimate is not None
            and result.external_point_estimate is not None
        ):
            assert result.delta_point_estimate == pytest.approx(
                result.external_point_estimate - result.reference_point_estimate
            )

    finding_a = {
        result.metric_name: result
        for result in report.results
        if result.label_name == "Finding A"
    }
    assert (
        finding_a["auroc"].delta_point_estimate < 0
        or finding_a["auprc"].delta_point_estimate < 0
    )
    assert (
        finding_a["brier_score"].delta_point_estimate > 0
        or finding_a["ece"].delta_point_estimate > 0
    )


def test_cross_dataset_bootstrap_is_deterministic_for_identical_seed(
    tmp_path: Path,
) -> None:
    first = _run_report(tmp_path / "first", seed=77)
    second = _run_report(tmp_path / "second", seed=77)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_different_seed_can_change_interval_values(tmp_path: Path) -> None:
    first = _run_report(tmp_path / "first", seed=11, metrics=("brier_score",))
    second = _run_report(tmp_path / "second", seed=22, metrics=("brier_score",))

    first_intervals = [
        (
            result.reference_ci_lower,
            result.reference_ci_upper,
            result.external_ci_lower,
            result.external_ci_upper,
            result.delta_ci_lower,
            result.delta_ci_upper,
        )
        for result in first.results
    ]
    second_intervals = [
        (
            result.reference_ci_lower,
            result.reference_ci_upper,
            result.external_ci_lower,
            result.external_ci_upper,
            result.delta_ci_lower,
            result.delta_ci_upper,
        )
        for result in second.results
    ]
    assert first_intervals != second_intervals


def test_reference_and_external_datasets_are_independently_resampled(
    tmp_path: Path,
) -> None:
    reference_predictions, reference_labels = _write_dataset(
        tmp_path,
        stem="reference",
        dataset_name="same-distribution-reference",
        scores_a=REFERENCE_A,
        scores_b=REFERENCE_B,
    )
    external_predictions, external_labels = _write_dataset(
        tmp_path,
        stem="external",
        dataset_name="same-distribution-external",
        scores_a=REFERENCE_A,
        scores_b=REFERENCE_B,
    )

    report = run_cross_dataset_bootstrap_from_files(
        reference_predictions_path=reference_predictions,
        reference_labels_csv_path=reference_labels,
        external_predictions_path=external_predictions,
        external_labels_csv_path=external_labels,
        iterations=50,
        seed=123,
        confidence_level=0.95,
        n_bins=4,
        metrics=("brier_score",),
    )

    result = report.results[0]
    assert result.delta_point_estimate == pytest.approx(0.0)
    assert result.delta_ci_lower < 0.0 or result.delta_ci_upper > 0.0


def test_patient_cluster_and_sample_bootstrap_selection(tmp_path: Path) -> None:
    patient_report = _run_report(tmp_path / "patient", iterations=20)
    assert patient_report.reference_resampling_unit == "patient"
    assert patient_report.external_resampling_unit == "patient"

    incomplete_patient_ids = tuple(
        None if index == 3 else f"patient-reference-{index}"
        for index in range(len(TARGETS_A))
    )
    sample_report = _run_report(
        tmp_path / "sample",
        reference_patient_ids=incomplete_patient_ids,
        iterations=20,
    )
    assert sample_report.reference_resampling_unit == "sample"
    assert sample_report.external_resampling_unit == "patient"


def test_invalid_discrimination_bootstrap_iterations_are_excluded(
    tmp_path: Path,
) -> None:
    targets_a = (1, 1, 1, 0)
    targets_b = (0, 0, 0, 1)
    scores_a = (0.9, 0.8, 0.7, 0.2)
    scores_b = (0.1, 0.2, 0.3, 0.8)
    patient_ids = tuple(None for _ in targets_a)
    reference_predictions, reference_labels = _write_dataset(
        tmp_path,
        stem="reference",
        dataset_name="reference-dataset",
        scores_a=scores_a,
        scores_b=scores_b,
        targets_a=targets_a,
        targets_b=targets_b,
        patient_ids=patient_ids,
    )
    external_predictions, external_labels = _write_dataset(
        tmp_path,
        stem="external",
        dataset_name="external-dataset",
        scores_a=scores_a,
        scores_b=scores_b,
        targets_a=targets_a,
        targets_b=targets_b,
        patient_ids=patient_ids,
    )

    report = run_cross_dataset_bootstrap_from_files(
        reference_predictions_path=reference_predictions,
        reference_labels_csv_path=reference_labels,
        external_predictions_path=external_predictions,
        external_labels_csv_path=external_labels,
        iterations=60,
        seed=5,
        metrics=("auroc", "auprc"),
    )

    for result in report.results:
        assert result.metric_name in {"auroc", "auprc"}
        assert result.reference_valid_iterations < 60
        assert result.external_valid_iterations < 60
        assert result.delta_valid_iterations < 60
        assert result.delta_valid_iterations <= result.reference_valid_iterations
        assert result.delta_valid_iterations <= result.external_valid_iterations


def test_validation_failures_are_clear(tmp_path: Path) -> None:
    reference_predictions, reference_labels = _write_dataset(
        tmp_path,
        stem="reference",
        dataset_name="reference-dataset",
    )
    external_predictions, external_labels = _write_dataset(
        tmp_path,
        stem="external",
        dataset_name="external-dataset",
    )

    with pytest.raises(ValueError, match="Unsupported cross-dataset bootstrap metric"):
        run_cross_dataset_bootstrap_from_files(
            reference_predictions_path=reference_predictions,
            reference_labels_csv_path=reference_labels,
            external_predictions_path=external_predictions,
            external_labels_csv_path=external_labels,
            metrics=("not_a_metric",),
        )

    other_predictions, other_labels = _write_dataset(
        tmp_path / "other",
        stem="other",
        dataset_name="other-dataset",
        labels=("Other Finding",),
    )
    with pytest.raises(ValueError, match="no shared labels"):
        run_cross_dataset_bootstrap_from_files(
            reference_predictions_path=reference_predictions,
            reference_labels_csv_path=reference_labels,
            external_predictions_path=other_predictions,
            external_labels_csv_path=other_labels,
        )

    different_model_predictions, different_model_labels = _write_dataset(
        tmp_path / "different-model",
        stem="external",
        dataset_name="external-dataset",
        model_name="other-model",
    )
    with pytest.raises(ValueError, match="same model_name"):
        run_cross_dataset_bootstrap_from_files(
            reference_predictions_path=reference_predictions,
            reference_labels_csv_path=reference_labels,
            external_predictions_path=different_model_predictions,
            external_labels_csv_path=different_model_labels,
        )


def test_exports_are_created_and_sanitized(tmp_path: Path) -> None:
    report = _run_report(tmp_path, iterations=25)
    outputs = write_cross_dataset_bootstrap_bundle(report, tmp_path / "outputs")

    assert outputs["json"].name == "cross_dataset_bootstrap_summary.json"
    assert outputs["csv"].name == "cross_dataset_bootstrap_summary.csv"
    assert outputs["json"].exists()
    assert outputs["csv"].exists()

    json_payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(outputs["csv"].open(encoding="utf-8", newline="")))
    assert len(csv_rows) == len(report.results)
    assert json_payload["delta_definition"] == "external_minus_reference"

    combined_output = (
        outputs["json"].read_text(encoding="utf-8")
        + outputs["csv"].read_text(encoding="utf-8")
        + json.dumps(report.model_dump(mode="json"))
    )
    assert str(tmp_path) not in combined_output
    assert "image_path" not in combined_output
    assert "reference_sample_" not in combined_output
    assert "external_sample_" not in combined_output
    assert "patient-" not in combined_output


def test_cli_success_and_code_2_on_invalid_input(tmp_path: Path) -> None:
    reference_predictions, reference_labels = _write_dataset(
        tmp_path,
        stem="reference",
        dataset_name="reference-dataset",
    )
    external_predictions, external_labels = _write_dataset(
        tmp_path,
        stem="external",
        dataset_name="external-dataset",
        scores_a=EXTERNAL_A,
        scores_b=EXTERNAL_B,
    )
    output_dir = tmp_path / "cli-output"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_cross_dataset_bootstrap_comparison.py"),
            "--reference-predictions",
            str(reference_predictions),
            "--reference-labels-csv",
            str(reference_labels),
            "--external-predictions",
            str(external_predictions),
            "--external-labels-csv",
            str(external_labels),
            "--output-dir",
            str(output_dir),
            "--iterations",
            "20",
            "--seed",
            "101",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["schema_version"] == "medshiftlab.cross_dataset_bootstrap.v1"
    assert summary["outputs_written"] == {
        "csv": "cross_dataset_bootstrap_summary.csv",
        "json": "cross_dataset_bootstrap_summary.json",
    }
    assert summary["delta_definition"] == "external_minus_reference"
    assert summary["real_inference_performed"] is False
    assert summary["clinical_validation_completed"] is False
    assert (output_dir / "cross_dataset_bootstrap_summary.json").exists()
    assert (output_dir / "cross_dataset_bootstrap_summary.csv").exists()

    invalid = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_cross_dataset_bootstrap_comparison.py"),
            "--reference-predictions",
            str(reference_predictions),
            "--reference-labels-csv",
            str(reference_labels),
            "--external-predictions",
            str(external_predictions),
            "--external-labels-csv",
            str(external_labels),
            "--output-dir",
            str(tmp_path / "invalid-output"),
            "--metrics",
            "bad_metric",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert invalid.returncode == 2
    assert "Cross-dataset bootstrap error:" in invalid.stderr
