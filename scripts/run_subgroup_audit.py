#!/usr/bin/env python3
"""Run subgroup/slice audit summaries for MedShiftLab calibrated evaluation.

This script evaluates aggregate and per-label threshold metrics across
user-provided subgroup columns on the evaluation split.

It is intended for exploratory local/manual analysis only. It does not perform
clinical validation, fairness validation, or external validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _metrics_for_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    if precision is None or sensitivity is None or (precision + sensitivity) == 0:
        f1 = None
    else:
        f1 = float(2.0 * precision * sensitivity / (precision + sensitivity))

    if sensitivity is None or specificity is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = float((sensitivity + specificity) / 2.0)

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def run_subgroup_audit(
    *,
    predictions_path: Path,
    labels_csv_path: Path,
    split_csv_path: Path,
    calibrated_threshold_eval_path: Path,
    metadata_csv_path: Path,
    subgroup_columns: list[str],
    output_json_path: Path,
    output_label_metrics_csv_path: Path,
    output_aggregate_csv_path: Path,
    split_name: str = "evaluation",
    notes: str = "",
) -> dict[str, Any]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(labels_csv_path)
    split = pd.read_csv(split_csv_path)
    calibrated_eval = json.loads(calibrated_threshold_eval_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(metadata_csv_path)

    if "sample_id" not in metadata.columns:
        raise ValueError("metadata_csv must contain a sample_id column.")

    missing_subgroups = sorted(set(subgroup_columns) - set(metadata.columns))
    if missing_subgroups:
        raise ValueError(f"metadata_csv is missing subgroup columns: {missing_subgroups}")

    label_names = list(predictions["label_names"])
    scores = _build_score_table(predictions)

    merged = (
        labels.merge(scores, on="sample_id", suffixes=("_true", "_score"), how="inner")
        .merge(split, on="sample_id", how="inner")
        .merge(metadata[["sample_id", *subgroup_columns]], on="sample_id", how="left")
    )

    evaluation = merged[merged["split"] == split_name].copy().reset_index(drop=True)
    if evaluation.empty:
        raise ValueError(f"No records found for split={split_name!r}.")

    selected_thresholds = {
        str(row["label"]): float(row["threshold"])
        for row in calibrated_eval["calibration_best_thresholds"]
    }

    missing_thresholds = sorted(set(label_names) - set(selected_thresholds))
    if missing_thresholds:
        raise ValueError(f"Missing selected thresholds for labels: {missing_thresholds}")

    label_metric_rows: list[dict[str, Any]] = []

    for subgroup_col in subgroup_columns:
        evaluation[subgroup_col] = evaluation[subgroup_col].fillna("Unknown").astype(str)

        for subgroup_value, group_df in evaluation.groupby(subgroup_col, dropna=False):
            subgroup_value = str(subgroup_value)
            n_records = int(len(group_df))

            for threshold_source in ["calibration_selected", "default_0.5"]:
                for label in label_names:
                    threshold = (
                        selected_thresholds[label]
                        if threshold_source == "calibration_selected"
                        else 0.5
                    )

                    y_true = group_df[f"{label}_true"].astype(int)
                    y_score = group_df[f"{label}_score"].astype(float)

                    n_pos = int((y_true == 1).sum())
                    n_neg = int((y_true == 0).sum())

                    metrics = _metrics_for_threshold(y_true, y_score, threshold)

                    label_metric_rows.append(
                        {
                            "subgroup_var": subgroup_col,
                            "subgroup_value": subgroup_value,
                            "threshold_source": threshold_source,
                            "label": label,
                            "threshold": threshold,
                            "n_records": n_records,
                            "n_pos": n_pos,
                            "n_neg": n_neg,
                            **metrics,
                        }
                    )

    label_df = pd.DataFrame(label_metric_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for (subgroup_var, subgroup_value, threshold_source), group in label_df.groupby(
        ["subgroup_var", "subgroup_value", "threshold_source"]
    ):
        aggregate_rows.append(
            {
                "subgroup_var": subgroup_var,
                "subgroup_value": subgroup_value,
                "threshold_source": threshold_source,
                "n_records": int(group["n_records"].iloc[0]),
                "n_labels": int(len(group)),
                "n_labels_with_f1": int(group["f1"].notna().sum()),
                "mean_f1": (
                    float(group["f1"].mean(skipna=True))
                    if group["f1"].notna().any()
                    else None
                ),
                "mean_sensitivity": (
                    float(group["sensitivity"].mean(skipna=True))
                    if group["sensitivity"].notna().any()
                    else None
                ),
                "mean_specificity": (
                    float(group["specificity"].mean(skipna=True))
                    if group["specificity"].notna().any()
                    else None
                ),
                "mean_balanced_accuracy": (
                    float(group["balanced_accuracy"].mean(skipna=True))
                    if group["balanced_accuracy"].notna().any()
                    else None
                ),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)

    label_df.to_csv(output_label_metrics_csv_path, index=False)
    aggregate_df.to_csv(output_aggregate_csv_path, index=False)

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

    subgroup_counts = {
        subgroup_col: evaluation[subgroup_col].value_counts(dropna=False).to_dict()
        for subgroup_col in subgroup_columns
    }

    payload = {
        "schema_version": "medshiftlab.subgroup_audit.v1",
        "source_predictions": predictions_path.name,
        "source_labels": labels_csv_path.name,
        "source_split": split_csv_path.name,
        "source_calibrated_threshold_eval": calibrated_threshold_eval_path.name,
        "source_metadata": metadata_csv_path.name,
        "dataset_name": dataset_name,
        "model_name": predictions.get("model_name"),
        "model_version": predictions.get("model_version"),
        "preprocessing_version": predictions.get("preprocessing_version"),
        "normalization": normalization,
        "split": split_name,
        "n_split_records": int(len(evaluation)),
        "subgroup_variables": subgroup_columns,
        "threshold_sources": ["calibration_selected", "default_0.5"],
        "subgroup_counts": subgroup_counts,
        "aggregate_metrics": aggregate_df.to_dict(orient="records"),
        "notes": notes,
    }

    output_json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--calibrated-threshold-eval", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--subgroup-columns", required=True, nargs="+")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-label-metrics-csv", required=True, type=Path)
    parser.add_argument("--output-aggregate-csv", required=True, type=Path)
    parser.add_argument("--split-name", default="evaluation")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_subgroup_audit(
        predictions_path=args.predictions,
        labels_csv_path=args.labels_csv,
        split_csv_path=args.split_csv,
        calibrated_threshold_eval_path=args.calibrated_threshold_eval,
        metadata_csv_path=args.metadata_csv,
        subgroup_columns=args.subgroup_columns,
        output_json_path=args.output_json,
        output_label_metrics_csv_path=args.output_label_metrics_csv,
        output_aggregate_csv_path=args.output_aggregate_csv,
        split_name=args.split_name,
        notes=args.notes,
    )

    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "dataset_name": payload["dataset_name"],
                "model_name": payload["model_name"],
                "split": payload["split"],
                "n_split_records": payload["n_split_records"],
                "subgroup_variables": payload["subgroup_variables"],
                "outputs_written": {
                    "json": str(args.output_json),
                    "label_metrics_csv": str(args.output_label_metrics_csv),
                    "aggregate_csv": str(args.output_aggregate_csv),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
