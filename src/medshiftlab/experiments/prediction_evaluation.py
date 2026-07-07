"""Evaluation orchestration for standardized prediction files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from medshiftlab.evaluation.report import EvaluationReport
from medshiftlab.evaluation.table import create_evaluation_report_from_rows
from medshiftlab.models.prediction import PredictionBatch, PredictionRecord
from medshiftlab.reporting.evaluation_export import (
    write_evaluation_label_metrics_csv,
    write_evaluation_report_json,
)
from medshiftlab.reporting.prediction_export import (
    read_prediction_batch_json,
    read_prediction_records_csv,
)


DEFAULT_EVALUATION_LIMIT = 128
MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE = 4096


class GroundTruthLabelRow(BaseModel):
    """Ground-truth labels for one prediction sample."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sample_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("sample_id", "image_id"),
    )

    @field_validator("sample_id")
    @classmethod
    def _strip_sample_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sample_id must not be empty")
        return value


class PredictionEvaluationConfig(BaseModel):
    """Validated configuration for one standardized prediction evaluation."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    n_bins: int = Field(default=10, gt=0)
    limit: int = Field(default=DEFAULT_EVALUATION_LIMIT, gt=0)
    allow_large_run: bool = False
    split: str | None = None
    uncertainty_strategy: str | None = None
    notes: str | None = None
    bootstrap_iters: int = Field(default=0, ge=0)
    figures_dir: Path | None = None


class PredictionEvaluationAccounting(BaseModel):
    """Success-path accounting for one prediction evaluation run."""

    model_config = ConfigDict(extra="forbid")

    total_prediction_records_loaded: int = Field(ge=0)
    total_label_rows_loaded: int = Field(ge=0)
    selected_prediction_records: int = Field(ge=0)
    selected_label_rows: int = Field(ge=0)
    evaluated_records: int = Field(ge=0)
    skipped_records: int = Field(ge=0)
    missing_predictions: int = Field(ge=0)
    missing_labels: int = Field(ge=0)
    duplicate_prediction_sample_ids: int = Field(ge=0)
    duplicate_label_sample_ids: int = Field(ge=0)
    label_mismatch_count: int = Field(ge=0)


class PredictionEvaluationResult(BaseModel):
    """Evaluation report plus run accounting for standardized predictions."""

    model_config = ConfigDict(extra="forbid")

    predictions_path: Path
    labels_csv_path: Path
    prediction_format: Literal["json", "csv"]
    report: EvaluationReport
    accounting: PredictionEvaluationAccounting
    support_status: dict[str, str]


def load_prediction_batch(path: str | Path) -> tuple[PredictionBatch, str]:
    """Load a standardized prediction batch from JSON or CSV."""

    prediction_path = Path(path)
    suffix = prediction_path.suffix.lower()
    if suffix == ".json":
        return read_prediction_batch_json(prediction_path), "json"
    if suffix == ".csv":
        return read_prediction_records_csv(prediction_path), "csv"
    raise ValueError(
        "Unsupported prediction file format. Expected .json or .csv for standardized "
        "prediction batches."
    )


def load_ground_truth_label_rows(
    path: str | Path,
    *,
    label_names: tuple[str, ...],
) -> list[GroundTruthLabelRow]:
    """Load and validate a CSV table with one row per sample_id."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ground-truth labels CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Ground-truth labels CSV must include a header row")

        fieldnames = set(reader.fieldnames)
        if not {"sample_id", "image_id"} & fieldnames:
            raise ValueError(
                "Ground-truth labels CSV must contain either 'sample_id' or 'image_id'"
            )
        missing_label_columns = [
            label_name for label_name in label_names if label_name not in fieldnames
        ]
        if missing_label_columns:
            raise ValueError(
                "Ground-truth labels CSV is missing required label column(s): "
                + ", ".join(missing_label_columns)
            )

        rows: list[GroundTruthLabelRow] = []
        for row_index, row in enumerate(reader):
            try:
                rows.append(GroundTruthLabelRow.model_validate(row))
            except Exception as error:
                raise ValueError(
                    f"Ground-truth labels row {row_index + 2} is invalid: {error}"
                ) from error
    if not rows:
        raise ValueError("Ground-truth labels CSV must contain at least one row")
    return rows


def run_prediction_batch_evaluation_from_files(
    *,
    predictions_path: str | Path,
    labels_csv_path: str | Path,
    config: PredictionEvaluationConfig,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> tuple[PredictionEvaluationResult, dict[str, Path]]:
    """Load standardized predictions and evaluate them against a label CSV."""

    prediction_batch, prediction_format = load_prediction_batch(predictions_path)
    label_rows = load_ground_truth_label_rows(
        labels_csv_path,
        label_names=prediction_batch.label_names,
    )
    result = run_prediction_batch_evaluation(
        predictions=prediction_batch,
        label_rows=label_rows,
        raw_label_rows_path=labels_csv_path,
        raw_predictions_path=predictions_path,
        prediction_format=prediction_format,
        config=config,
    )

    written_outputs: dict[str, Path] = {}
    if output_json is not None:
        written_outputs["json"] = write_evaluation_report_json(
            result.report,
            output_json,
        )
    if output_csv is not None:
        written_outputs["csv"] = write_evaluation_label_metrics_csv(
            result.report,
            output_csv,
        )
    return result, written_outputs


def run_prediction_batch_evaluation(
    *,
    predictions: PredictionBatch,
    label_rows: list[GroundTruthLabelRow],
    raw_label_rows_path: str | Path,
    raw_predictions_path: str | Path,
    prediction_format: str,
    config: PredictionEvaluationConfig,
) -> PredictionEvaluationResult:
    """Evaluate a standardized prediction batch against a strict label cohort."""

    if config.limit > MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE and not config.allow_large_run:
        raise ValueError(
            f"Evaluation limit {config.limit} exceeds the safe threshold of "
            f"{MAX_SAFE_EVALUATION_LIMIT_WITHOUT_OVERRIDE}. Re-run with "
            "--allow-large-run only for an explicit local/manual evaluation."
        )
    if config.bootstrap_iters > 0:
        raise NotImplementedError(
            "Bootstrap confidence intervals are not implemented in the current "
            "evaluation package. Keep bootstrap-iters at 0 for now."
        )
    if config.figures_dir is not None:
        raise NotImplementedError(
            "Calibration plot export is not implemented in the current evaluation "
            "package. Omit --figures-dir for now."
        )

    total_prediction_records_loaded = len(predictions.records)
    selected_predictions = predictions.records[: config.limit]
    skipped_records = total_prediction_records_loaded - len(selected_predictions)

    duplicate_prediction_ids = _count_duplicates(
        [record.sample_id for record in selected_predictions]
    )
    if duplicate_prediction_ids:
        raise ValueError(
            f"Selected prediction records contain duplicate sample_id values: "
            f"{duplicate_prediction_ids}"
        )

    label_rows_by_id: dict[str, GroundTruthLabelRow] = {}
    duplicate_label_ids = 0
    for row in label_rows:
        if row.sample_id in label_rows_by_id:
            duplicate_label_ids += 1
            continue
        label_rows_by_id[row.sample_id] = row
    if duplicate_label_ids:
        raise ValueError(
            f"Ground-truth labels CSV contains duplicate sample_id values: "
            f"{duplicate_label_ids}"
        )

    selected_prediction_ids = {record.sample_id for record in selected_predictions}
    label_ids = set(label_rows_by_id)
    missing_labels = len(selected_prediction_ids - label_ids)
    missing_predictions = len(label_ids - selected_prediction_ids)
    if missing_labels or missing_predictions:
        raise ValueError(
            "Prediction/label sample_id mismatch: "
            f"missing_labels={missing_labels}, missing_predictions={missing_predictions}"
        )

    dataset_names = {record.dataset_name for record in selected_predictions}
    if len(dataset_names) != 1:
        raise ValueError(
            "Prediction batch must contain exactly one dataset_name for evaluation"
        )

    evaluation_rows: list[dict[str, object]] = []
    label_mismatch_count = 0
    for record in selected_predictions:
        label_row = label_rows_by_id[record.sample_id]
        row: dict[str, object] = {
            "sample_id": record.sample_id,
            "image_id": record.sample_id,
            "dataset_name": record.dataset_name,
            "model_name": predictions.model_name,
        }
        for label_name, probability in zip(
            record.label_names,
            record.probabilities,
            strict=True,
        ):
            if label_name not in label_row.model_extra:
                label_mismatch_count += 1
                raise ValueError(
                    f"Ground-truth label row for sample_id {record.sample_id!r} "
                    f"is missing label {label_name!r}"
                )
            row[f"true_{label_name}"] = label_row.model_extra[label_name]
            row[f"score_{label_name}"] = probability
        evaluation_rows.append(row)

    report = create_evaluation_report_from_rows(
        rows=evaluation_rows,
        labels=predictions.label_names,
        dataset_name=next(iter(dataset_names)),
        model_name=predictions.model_name,
        split=config.split,
        uncertainty_strategy=(
            predictions.uncertainty_strategy
            if config.uncertainty_strategy is None
            else config.uncertainty_strategy
        ),
        threshold=config.threshold,
        n_bins=config.n_bins,
        notes=config.notes,
    )

    accounting = PredictionEvaluationAccounting(
        total_prediction_records_loaded=total_prediction_records_loaded,
        total_label_rows_loaded=len(label_rows),
        selected_prediction_records=len(selected_predictions),
        selected_label_rows=len(label_rows),
        evaluated_records=len(evaluation_rows),
        skipped_records=skipped_records,
        missing_predictions=0,
        missing_labels=0,
        duplicate_prediction_sample_ids=0,
        duplicate_label_sample_ids=0,
        label_mismatch_count=label_mismatch_count,
    )

    return PredictionEvaluationResult(
        predictions_path=Path(raw_predictions_path),
        labels_csv_path=Path(raw_label_rows_path),
        prediction_format=prediction_format if prediction_format in {"json", "csv"} else "json",
        report=report,
        accounting=accounting,
        support_status={
            "bootstrap_ci": "not_implemented",
            "calibration_curve_data": "not_implemented",
            "calibration_plot_export": "not_implemented",
            "confusion_matrix": "per_label_counts_in_metrics",
        },
    )


def _count_duplicates(sample_ids: list[str]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for sample_id in sample_ids:
        if sample_id in seen:
            duplicates += 1
            continue
        seen.add(sample_id)
    return duplicates
