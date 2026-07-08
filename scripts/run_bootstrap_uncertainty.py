#!/usr/bin/env python3
"""Run bootstrap uncertainty summaries for MedShiftLab calibrated evaluation.

This script resamples the evaluation split with replacement and summarizes
aggregate metric uncertainty for calibration-selected thresholds versus the
default 0.5 threshold.

It is intended for exploratory local/manual analysis only. It does not perform
clinical validation and must not be used to claim diagnostic performance.
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
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
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


def _aggregate_for_sample(
    *,
    df: pd.DataFrame,
    label_names: list[str],
    selected_thresholds: dict[str, float],
    threshold_source: str,
) -> dict[str, float | str]:
    per_label = []

    for label in label_names:
        threshold = selected_thresholds[label] if threshold_source == "calibration_selected" else 0.5

        y_true = df[f"{label}_true"].astype(int)
        y_score = df[f"{label}_score"].astype(float)

        per_label.append(_metrics_for_threshold(y_true, y_score, threshold))

    return {
        "threshold_source": threshold_source,
        "mean_f1": float(np.mean([m["f1"] for m in per_label])),
        "mean_sensitivity": float(np.mean([m["sensitivity"] for m in per_label])),
        "mean_specificity": float(np.mean([m["specificity"] for m in per_label])),
        "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in per_label])),
    }


def run_bootstrap_uncertainty(
    *,
    predictions_path: Path,
    labels_csv_path: Path,
    split_csv_path: Path,
    calibrated_threshold_eval_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 20260708,
    notes: str = "",
) -> dict[str, Any]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(labels_csv_path)
    split = pd.read_csv(split_csv_path)
    calibrated_eval = json.loads(calibrated_threshold_eval_path.read_text(encoding="utf-8"))

    label_names = list(predictions["label_names"])
    scores = _build_score_table(predictions)

    merged = (
        labels.merge(scores, on="sample_id", suffixes=("_true", "_score"), how="inner")
        .merge(split, on="sample_id", how="inner")
    )

    evaluation = merged[merged["split"] == "evaluation"].copy().reset_index(drop=True)
    if evaluation.empty:
        raise ValueError("No evaluation split records found.")

    selected_thresholds = {
        str(row["label"]): float(row["threshold"])
        for row in calibrated_eval["calibration_best_thresholds"]
    }

    missing_thresholds = sorted(set(label_names) - set(selected_thresholds))
    if missing_thresholds:
        raise ValueError(f"Missing selected thresholds for labels: {missing_thresholds}")

    point_rows: list[dict[str, Any]] = []
    for source in ["calibration_selected", "default_0.5"]:
        row = _aggregate_for_sample(
            df=evaluation,
            label_names=label_names,
            selected_thresholds=selected_thresholds,
            threshold_source=source,
        )
        row["sample_type"] = "point_estimate"
        row["bootstrap_iteration"] = -1
        point_rows.append(row)

    rng = np.random.default_rng(bootstrap_seed)
    boot_rows: list[dict[str, Any]] = []

    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(evaluation), size=len(evaluation))
        boot_df = evaluation.iloc[indices].reset_index(drop=True)

        for source in ["calibration_selected", "default_0.5"]:
            row = _aggregate_for_sample(
                df=boot_df,
                label_names=label_names,
                selected_thresholds=selected_thresholds,
                threshold_source=source,
            )
            row["sample_type"] = "bootstrap"
            row["bootstrap_iteration"] = iteration
            boot_rows.append(row)

    result_df = pd.DataFrame(point_rows + boot_rows)

    summary_rows: list[dict[str, Any]] = []
    metrics = [
        "mean_f1",
        "mean_sensitivity",
        "mean_specificity",
        "mean_balanced_accuracy",
    ]

    for source in ["calibration_selected", "default_0.5"]:
        subset = result_df[
            (result_df["threshold_source"] == source)
            & (result_df["sample_type"] == "bootstrap")
        ]
        point = result_df[
            (result_df["threshold_source"] == source)
            & (result_df["sample_type"] == "point_estimate")
        ].iloc[0]

        for metric in metrics:
            values = subset[metric].astype(float).to_numpy()
            summary_rows.append(
                {
                    "threshold_source": source,
                    "metric": metric,
                    "point_estimate": float(point[metric]),
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_lower_2_5": float(np.percentile(values, 2.5)),
                    "ci_median_50": float(np.percentile(values, 50.0)),
                    "ci_upper_97_5": float(np.percentile(values, 97.5)),
                    "n_bootstrap": int(n_bootstrap),
                    "n_evaluation_records": int(len(evaluation)),
                }
            )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv_path, index=False)

    first_record = predictions["records"][0] if predictions.get("records") else {}
    dataset_name = (
        calibrated_eval.get("dataset_name")
        or predictions.get("dataset_name")
        or first_record.get("dataset_name")
        or "unknown"
    )

    normalization = (
        predictions.get("preprocessing_config", {})
        .get("image_preprocessing", {})
        .get("normalization")
    )

    payload = {
        "schema_version": "medshiftlab.bootstrap_uncertainty.v1",
        "source_predictions": predictions_path.name,
        "source_labels": labels_csv_path.name,
        "source_split": split_csv_path.name,
        "source_calibrated_threshold_eval": calibrated_threshold_eval_path.name,
        "dataset_name": dataset_name,
        "model_name": predictions.get("model_name"),
        "model_version": predictions.get("model_version"),
        "preprocessing_version": predictions.get("preprocessing_version"),
        "normalization": normalization,
        "n_evaluation_records": int(len(evaluation)),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
        "metrics": summary_rows,
        "notes": notes,
    }

    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--calibrated-threshold-eval", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--n-bootstrap", default=1000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260708, type=int)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_bootstrap_uncertainty(
        predictions_path=args.predictions,
        labels_csv_path=args.labels_csv,
        split_csv_path=args.split_csv,
        calibrated_threshold_eval_path=args.calibrated_threshold_eval,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        notes=args.notes,
    )

    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "dataset_name": payload["dataset_name"],
                "model_name": payload["model_name"],
                "n_evaluation_records": payload["n_evaluation_records"],
                "n_bootstrap": payload["n_bootstrap"],
                "bootstrap_seed": payload["bootstrap_seed"],
                "outputs_written": {
                    "json": str(args.output_json),
                    "csv": str(args.output_csv),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
