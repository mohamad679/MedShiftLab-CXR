"""File export utilities for standardized MedShiftLab-CXR predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from medshiftlab.models.prediction import PredictionBatch, PredictionRecord


_PREDICTION_RECORDS_HEADER = (
    "schema_version",
    "created_at",
    "model_name",
    "model_version",
    "adapter_name",
    "preprocessing_version",
    "preprocessing_config",
    "uncertainty_strategy",
    "run_metadata",
    "sample_id",
    "patient_id",
    "study_id",
    "dataset_name",
    "image_path",
    "label_names",
    "probabilities",
    "logits",
    "thresholds",
    "thresholded_predictions",
    "record_metadata",
)


def write_prediction_batch_json(
    predictions: PredictionBatch, output_path: str | Path
) -> Path:
    """Write a complete prediction batch as pretty-printed JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = predictions.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_prediction_batch_json(input_path: str | Path) -> PredictionBatch:
    """Load and validate a prediction batch from JSON."""

    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PredictionBatch.model_validate(payload)


def write_prediction_records_csv(
    predictions: PredictionBatch, output_path: str | Path
) -> Path:
    """Write one stable CSV row per prediction record."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_PREDICTION_RECORDS_HEADER)
        writer.writeheader()
        for record in predictions.records:
            writer.writerow(_batch_row(predictions, record))
    return path


def read_prediction_records_csv(input_path: str | Path) -> PredictionBatch:
    """Load and validate a prediction batch from record-wise CSV rows."""

    path = Path(input_path)
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("prediction CSV must include a header row")
        missing_columns = [
            column
            for column in _PREDICTION_RECORDS_HEADER
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "prediction CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        rows = list(reader)

    if not rows:
        raise ValueError("prediction CSV must contain at least one record row")

    first_row = rows[0]
    batch = PredictionBatch(
        schema_version=first_row["schema_version"],
        created_at=first_row["created_at"],
        model_name=first_row["model_name"],
        model_version=first_row["model_version"],
        adapter_name=first_row["adapter_name"],
        preprocessing_version=first_row["preprocessing_version"],
        preprocessing_config=_read_json_value(
            first_row["preprocessing_config"],
            expected_type=dict,
        ),
        uncertainty_strategy=first_row["uncertainty_strategy"] or None,
        run_metadata=_read_json_value(first_row["run_metadata"], expected_type=dict),
        label_names=_read_json_value(first_row["label_names"], expected_type=list),
        records=[
            _row_to_prediction_record(row, batch_model_name=first_row["model_name"])
            for row in rows
        ],
    )

    for row in rows[1:]:
        _ensure_matching_batch_metadata(first_row, row)
    return batch


def _batch_row(
    predictions: PredictionBatch,
    record: PredictionRecord,
) -> dict[str, str]:
    return {
        "schema_version": predictions.schema_version,
        "created_at": predictions.created_at.isoformat(),
        "model_name": predictions.model_name,
        "model_version": predictions.model_version,
        "adapter_name": predictions.adapter_name,
        "preprocessing_version": predictions.preprocessing_version,
        "preprocessing_config": json.dumps(predictions.preprocessing_config, sort_keys=True),
        "uncertainty_strategy": predictions.uncertainty_strategy or "",
        "run_metadata": json.dumps(predictions.run_metadata, sort_keys=True),
        "sample_id": record.sample_id,
        "patient_id": record.patient_id or "",
        "study_id": record.study_id or "",
        "dataset_name": record.dataset_name,
        "image_path": record.image_path or "",
        "label_names": json.dumps(list(record.label_names)),
        "probabilities": json.dumps(list(record.probabilities)),
        "logits": _json_or_blank(record.logits),
        "thresholds": _json_or_blank(record.thresholds),
        "thresholded_predictions": _json_or_blank(record.thresholded_predictions),
        "record_metadata": json.dumps(record.metadata, sort_keys=True),
    }


def _row_to_prediction_record(
    row: dict[str, str],
    *,
    batch_model_name: str,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=row["sample_id"],
        patient_id=row["patient_id"] or None,
        study_id=row["study_id"] or None,
        dataset_name=row["dataset_name"],
        image_path=row["image_path"] or None,
        model_name=batch_model_name,
        label_names=_read_json_value(row["label_names"], expected_type=list),
        probabilities=_read_json_value(row["probabilities"], expected_type=list),
        logits=_read_optional_json_value(row["logits"], expected_type=list),
        thresholds=_read_optional_json_value(
            row["thresholds"],
            expected_type=(float, int, list),
        ),
        thresholded_predictions=_read_optional_json_value(
            row["thresholded_predictions"],
            expected_type=list,
        ),
        metadata=_read_json_value(row["record_metadata"], expected_type=dict),
    )


def _ensure_matching_batch_metadata(
    first_row: dict[str, str],
    row: dict[str, str],
) -> None:
    for field_name in (
        "schema_version",
        "created_at",
        "model_name",
        "model_version",
        "adapter_name",
        "preprocessing_version",
        "preprocessing_config",
        "uncertainty_strategy",
        "run_metadata",
        "label_names",
    ):
        if row[field_name] != first_row[field_name]:
            raise ValueError(
                f"prediction CSV contains inconsistent batch metadata for {field_name}"
            )


def _json_or_blank(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return json.dumps(list(value))
    return json.dumps(value)


def _read_json_value(value: str, *, expected_type: type | tuple[type, ...]) -> object:
    loaded = json.loads(value)
    if not isinstance(loaded, expected_type):
        expected = (
            ", ".join(type_.__name__ for type_ in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ValueError(f"prediction export expected JSON type {expected}")
    return loaded


def _read_optional_json_value(
    value: str,
    *,
    expected_type: type | tuple[type, ...],
) -> object | None:
    if not value:
        return None
    loaded = _read_json_value(value, expected_type=expected_type)
    if isinstance(loaded, list):
        return tuple(loaded)
    if isinstance(loaded, int):
        return float(loaded)
    return loaded
