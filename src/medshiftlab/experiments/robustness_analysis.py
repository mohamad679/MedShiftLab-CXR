"""Manual-only orchestration for Phase 10 robustness analysis artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.evaluation.report import EvaluationReport
from medshiftlab.evaluation.robustness import (
    RobustnessAnalysisReport,
    analyze_subgroups,
    bootstrap_label_metric_intervals,
    build_calibration_summaries,
    summarize_failure_cases,
)
from medshiftlab.experiments.prediction_evaluation import (
    DEFAULT_EVALUATION_LIMIT,
    MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE,
    PredictionEvaluationConfig,
    load_ground_truth_label_rows,
    load_prediction_batch,
    run_prediction_batch_evaluation_from_files,
)


class RobustnessAnalysisConfig(BaseModel):
    """Validated run controls for a bounded local robustness analysis."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    n_bins: int = Field(default=10, gt=0)
    limit: int = Field(default=DEFAULT_EVALUATION_LIMIT, gt=0)
    allow_large_run: bool = False
    split: str | None = None
    uncertainty_strategy: str | None = None
    bootstrap_iters: int = Field(default=0, ge=0)
    bootstrap_metrics: tuple[str, ...] = ("auroc", "brier_score", "ece")
    seed: int = 2026
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    subgroup_columns: tuple[str, ...] = (
        "dataset_name",
        "split",
        "uncertainty_strategy",
    )
    minimum_subgroup_size: int = Field(default=20, gt=0)
    degradation_metric: str = "auroc"
    degradation_threshold: float | None = Field(default=None, ge=0.0)
    maximum_ece: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_brier_score: float | None = Field(default=None, ge=0.0, le=1.0)


def run_robustness_analysis_from_files(
    *,
    predictions_path: str | Path,
    labels_csv_path: str | Path,
    config: RobustnessAnalysisConfig,
    metadata_csv_path: str | Path | None = None,
    baseline_evaluation_json: str | Path | None = None,
) -> RobustnessAnalysisReport:
    """Analyze standardized predictions without running inference or fitting."""

    if config.limit > MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE and not config.allow_large_run:
        raise ValueError(
            f"Analysis limit {config.limit} exceeds the safe threshold of "
            f"{MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE}. Re-run with "
            "--allow-large-run only for an explicit local/manual analysis."
        )

    evaluation_result, _ = run_prediction_batch_evaluation_from_files(
        predictions_path=predictions_path,
        labels_csv_path=labels_csv_path,
        config=PredictionEvaluationConfig(
            threshold=config.threshold,
            n_bins=config.n_bins,
            limit=config.limit,
            allow_large_run=config.allow_large_run,
            split=config.split,
            uncertainty_strategy=config.uncertainty_strategy,
        ),
    )
    _reject_absolute_identifier(
        evaluation_result.report.metadata.dataset_name, "dataset_name"
    )
    _reject_absolute_identifier(
        evaluation_result.report.metadata.model_name, "model_name"
    )
    prediction_batch, _ = load_prediction_batch(predictions_path)
    label_rows = load_ground_truth_label_rows(
        labels_csv_path,
        label_names=prediction_batch.label_names,
    )
    labels_by_id = {row.sample_id: row for row in label_rows}
    metadata_by_id = (
        _load_metadata_rows(metadata_csv_path) if metadata_csv_path is not None else {}
    )

    rows: list[dict[str, object]] = []
    patient_ids: list[str | None] = []
    for record in prediction_batch.records[: config.limit]:
        label_row = labels_by_id[record.sample_id]
        metadata = metadata_by_id.get(record.sample_id, {})
        row: dict[str, object] = {
            "sample_id": record.sample_id,
            "dataset_name": record.dataset_name,
            "split": config.split,
            "uncertainty_strategy": (
                config.uncertainty_strategy or prediction_batch.uncertainty_strategy
            ),
        }
        for key, value in metadata.items():
            if key not in {"sample_id", "image_id"}:
                row[key] = value
        for label_name, score in zip(
            record.label_names, record.probabilities, strict=True
        ):
            row[f"true_{label_name}"] = label_row.model_extra[label_name]
            row[f"score_{label_name}"] = score
        rows.append(row)
        patient_value = record.patient_id or metadata.get("patient_id")
        patient_ids.append(str(patient_value) if patient_value else None)

    subgroup_report = analyze_subgroups(
        rows,
        labels=prediction_batch.label_names,
        subgroup_columns=config.subgroup_columns,
        minimum_subgroup_size=config.minimum_subgroup_size,
        threshold=config.threshold,
        n_bins=config.n_bins,
    )
    calibration = build_calibration_summaries(
        rows,
        labels=prediction_batch.label_names,
        n_bins=config.n_bins,
    )

    bootstrap_intervals = []
    if config.bootstrap_iters:
        for label_name in prediction_batch.label_names:
            bootstrap_intervals.extend(
                bootstrap_label_metric_intervals(
                    label_name,
                    [row[f"true_{label_name}"] for row in rows],
                    [row[f"score_{label_name}"] for row in rows],
                    metrics=config.bootstrap_metrics,
                    iterations=config.bootstrap_iters,
                    seed=config.seed,
                    threshold=config.threshold,
                    n_bins=config.n_bins,
                    confidence_level=config.confidence_level,
                    patient_ids=patient_ids if all(patient_ids) else None,
                )
            )

    baseline = (
        _load_evaluation_report(baseline_evaluation_json)
        if baseline_evaluation_json is not None
        else None
    )
    failure_cases = summarize_failure_cases(
        evaluation_result.report,
        subgroup_report=subgroup_report,
        baseline_report=baseline,
        degradation_metric=config.degradation_metric,
        degradation_threshold=config.degradation_threshold,
        maximum_ece=config.maximum_ece,
        maximum_brier_score=config.maximum_brier_score,
    )

    return RobustnessAnalysisReport(
        dataset_name=evaluation_result.report.metadata.dataset_name,
        model_name=evaluation_result.report.metadata.model_name,
        evaluated_records=evaluation_result.accounting.evaluated_records,
        calibration=calibration,
        bootstrap_intervals=bootstrap_intervals,
        subgroup_analysis=subgroup_report,
        failure_cases=failure_cases,
        support_status={
            "calibration_bin_data": "implemented",
            "calibration_curve_csv": "implemented",
            "calibration_plot_export": "implemented_optional",
            "bootstrap_ci": (
                "implemented_not_requested"
                if config.bootstrap_iters == 0
                else "implemented"
            ),
            "bootstrap_resampling": (
                "patient" if config.bootstrap_iters and all(patient_ids) else "sample"
            ),
            "subgroup_analysis": "implemented",
            "failure_case_flags": "implemented_non_clinical",
        },
    )


def _load_metadata_rows(path: str | Path) -> dict[str, dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Subgroup metadata CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"sample_id", "image_id"} & set(reader.fieldnames):
            raise ValueError("Subgroup metadata CSV must contain sample_id or image_id")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            sample_id = (row.get("sample_id") or row.get("image_id") or "").strip()
            if not sample_id:
                raise ValueError("Subgroup metadata contains an empty sample identifier")
            if sample_id in rows:
                raise ValueError(f"Subgroup metadata contains duplicate sample_id: {sample_id!r}")
            rows[sample_id] = row
    return rows


def _load_evaluation_report(path: str | Path) -> EvaluationReport:
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Baseline evaluation JSON not found: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Baseline evaluation JSON is invalid") from error
    return EvaluationReport.model_validate(payload)


def _reject_absolute_identifier(value: str, field_name: str) -> None:
    if value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError(f"{field_name} must not contain an absolute path")
