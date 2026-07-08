#!/usr/bin/env python3
"""Validate VinDr/VinBigData prediction CSV schema.

This script validates prediction files against prepared manifest/label inputs.
It does not generate predictions, does not run model inference, and does not
compute metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def _read_csv(path: Path, *, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"{name} is empty: {path}")
    return data


def validate_prediction_schema(
    *,
    predictions_csv: Path,
    manifest_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    target_labels: list[str] | None = None,
    require_exact_samples: bool = True,
) -> dict[str, Any]:
    target_labels = target_labels or DEFAULT_TARGET_LABELS

    predictions = _read_csv(predictions_csv, name="predictions CSV")
    manifest = _read_csv(manifest_csv, name="manifest CSV")
    labels = _read_csv(labels_csv, name="labels CSV")

    required_prediction_columns = {"sample_id", *target_labels}
    missing_prediction_columns = sorted(required_prediction_columns - set(predictions.columns))
    if missing_prediction_columns:
        raise ValueError(f"predictions CSV is missing required columns: {missing_prediction_columns}")

    for label in target_labels:
        values = pd.to_numeric(predictions[label], errors="coerce")
        if values.isna().any():
            raise ValueError(f"prediction column contains non-numeric values: {label}")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"prediction column must be within [0, 1]: {label}")

    manifest_samples = set(manifest["sample_id"].astype(str))
    label_samples = set(labels["sample_id"].astype(str))
    prediction_samples = set(predictions["sample_id"].astype(str))

    if len(prediction_samples) != len(predictions):
        raise ValueError("predictions CSV contains duplicate sample_id values")

    missing_from_labels = sorted(prediction_samples - label_samples)
    missing_from_manifest = sorted(prediction_samples - manifest_samples)
    if missing_from_labels:
        raise ValueError(f"predictions contain samples not present in labels: {len(missing_from_labels)}")
    if missing_from_manifest:
        raise ValueError(f"predictions contain samples not present in manifest: {len(missing_from_manifest)}")

    if require_exact_samples:
        missing_predictions = sorted((manifest_samples & label_samples) - prediction_samples)
        if missing_predictions:
            raise ValueError(f"predictions are missing expected samples: {len(missing_predictions)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "vindr_prediction_schema_validation_summary.json"

    payload = {
        "schema_version": "medshiftlab.vindr_prediction_schema_validation.v1",
        "claim_level": {
            "predictions_generated_by_this_script": False,
            "model_inference_completed": False,
            "metrics_completed": False,
            "external_validation_completed": False,
            "clinical_validation_completed": False,
        },
        "prediction_summary": {
            "n_prediction_rows": int(len(predictions)),
            "target_labels": target_labels,
            "require_exact_samples": bool(require_exact_samples),
            "min_prediction": {
                label: float(pd.to_numeric(predictions[label], errors="raise").min())
                for label in target_labels
            },
            "max_prediction": {
                label: float(pd.to_numeric(predictions[label], errors="raise").max())
                for label in target_labels
            },
        },
        "reference_summary": {
            "n_manifest_rows": int(len(manifest)),
            "n_label_rows": int(len(labels)),
            "n_manifest_label_overlap": int(len(manifest_samples & label_samples)),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "metrics_json": None,
        },
        "notes": (
            "Prediction schema validation only. This script does not create predictions, "
            "does not run model inference, and does not compute external-validation metrics."
        ),
    }

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-labels", nargs="*", default=DEFAULT_TARGET_LABELS)
    parser.add_argument("--allow-subset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = validate_prediction_schema(
        predictions_csv=args.predictions_csv,
        manifest_csv=args.manifest_csv,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        target_labels=list(args.target_labels),
        require_exact_samples=not args.allow_subset,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
