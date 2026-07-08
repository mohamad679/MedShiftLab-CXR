#!/usr/bin/env python3
"""Run a per-label threshold sweep for MedShiftLab prediction outputs.

This script is intended for local/manual evaluation outputs. It does not
perform clinical validation and should not be used to claim diagnostic
performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_for_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * sensitivity, precision + sensitivity)
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _build_score_table(predictions: dict[str, Any]) -> pd.DataFrame:
    label_names = predictions["label_names"]
    rows: list[dict[str, Any]] = []

    for record in predictions["records"]:
        row: dict[str, Any] = {"sample_id": record["sample_id"]}
        record_labels = record.get("label_names", label_names)

        for label, probability in zip(record_labels, record["probabilities"]):
            row[str(label)] = float(probability)

        rows.append(row)

    return pd.DataFrame(rows)


def _threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if stop < start:
        raise ValueError("threshold stop must be greater than or equal to start")

    values = np.arange(start, stop + (step / 2.0), step)
    return np.round(values, 10)


def run_threshold_sweep(
    *,
    predictions_path: Path,
    labels_csv_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    output_best_csv_path: Path,
    threshold_start: float = 0.0,
    threshold_stop: float = 1.0,
    threshold_step: float = 0.01,
    notes: str = "",
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(labels_csv_path)

    label_names = list(predictions["label_names"])
    scores = _build_score_table(predictions)

    merged = labels.merge(scores, on="sample_id", suffixes=("_true", "_score"), how="inner")
    if merged.empty:
        raise ValueError("No matched sample_id rows between labels and predictions.")

    thresholds = _threshold_grid(threshold_start, threshold_stop, threshold_step)

    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for label in label_names:
        true_col = f"{label}_true"
        score_col = f"{label}_score"

        if true_col not in merged.columns or score_col not in merged.columns:
            raise ValueError(f"Missing label columns for {label!r}")

        y_true = merged[true_col].astype(float)
        y_score = merged[score_col].astype(float)

        valid = y_true.isin([0.0, 1.0]) & y_score.notna()
        y_true = y_true[valid].astype(int)
        y_score = y_score[valid]

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        label_rows: list[dict[str, Any]] = []

        for threshold in thresholds:
            metrics = _metrics_for_threshold(y_true, y_score, float(threshold))
            row = {
                "label": label,
                "threshold": float(threshold),
                "n": int(len(y_true)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                **metrics,
            }
            label_rows.append(row)
            all_rows.append(row)

        best = sorted(
            label_rows,
            key=lambda row: (
                row["f1"],
                row["balanced_accuracy"],
                row["specificity"],
                -abs(row["threshold"] - 0.5),
                -row["threshold"],
            ),
            reverse=True,
        )[0]
        best_rows.append(best)

    sweep_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_best_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_df.to_csv(output_csv_path, index=False)
    best_df.to_csv(output_best_csv_path, index=False)

    first_record = predictions["records"][0] if predictions.get("records") else {}
    dataset_name = (
        predictions.get("dataset_name")
        or first_record.get("dataset_name")
        or "unknown"
    )

    normalization = (
        predictions.get("preprocessing_config", {})
        .get("image_preprocessing", {})
        .get("normalization")
    )

    payload = {
        "schema_version": "medshiftlab.threshold_sweep.v1",
        "source_predictions": predictions_path.name,
        "source_labels": labels_csv_path.name,
        "dataset_name": dataset_name,
        "model_name": predictions.get("model_name"),
        "model_version": predictions.get("model_version"),
        "preprocessing_version": predictions.get("preprocessing_version"),
        "normalization": normalization,
        "n_records": int(len(merged)),
        "label_names": label_names,
        "threshold_grid": {
            "start": threshold_start,
            "stop": threshold_stop,
            "step": threshold_step,
        },
        "selection_rule": [
            "maximize_f1",
            "maximize_balanced_accuracy",
            "maximize_specificity",
            "minimize_distance_to_0.5",
            "prefer_lower_threshold",
        ],
        "best_thresholds": best_rows,
        "notes": notes,
    }

    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-best-csv", required=True, type=Path)
    parser.add_argument("--threshold-start", default=0.0, type=float)
    parser.add_argument("--threshold-stop", default=1.0, type=float)
    parser.add_argument("--threshold-step", default=0.01, type=float)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_threshold_sweep(
        predictions_path=args.predictions,
        labels_csv_path=args.labels_csv,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        output_best_csv_path=args.output_best_csv,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
        notes=args.notes,
    )

    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "dataset_name": payload["dataset_name"],
                "model_name": payload["model_name"],
                "n_records": payload["n_records"],
                "n_labels": len(payload["label_names"]),
                "outputs_written": {
                    "json": str(args.output_json),
                    "csv": str(args.output_csv),
                    "best_csv": str(args.output_best_csv),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
