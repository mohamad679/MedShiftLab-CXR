#!/usr/bin/env python3
"""Run calibration/evaluation split threshold selection for MedShiftLab outputs.

This script selects per-label thresholds on a deterministic calibration split
and evaluates those selected thresholds on a separate evaluation split.

It is intended for exploratory local/manual analysis only. It does not perform
clinical validation and must not be used to claim diagnostic performance.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if stop < start:
        raise ValueError("threshold stop must be greater than or equal to start")
    return np.round(np.arange(start, stop + (step / 2.0), step), 10)


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


def _make_balanced_split(
    merged: pd.DataFrame,
    label_names: list[str],
    calibration_fraction: float,
) -> pd.DataFrame:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    target_cal_size = int(round(len(merged) * calibration_fraction))

    work = merged.copy()
    work["_n_pos"] = work[[f"{label}_true" for label in label_names]].sum(axis=1)
    work["_hash"] = work["sample_id"].map(
        lambda sample_id: int(hashlib.sha256(str(sample_id).encode()).hexdigest(), 16)
    )
    work = work.sort_values(["_n_pos", "_hash"], ascending=[False, True]).reset_index(drop=True)

    total_pos = {
        label: int((work[f"{label}_true"] == 1).sum())
        for label in label_names
    }
    target_pos = {
        label: total_pos[label] * calibration_fraction
        for label in label_names
    }

    cal_ids: list[str] = []
    eval_ids: list[str] = []
    cal_pos = {label: 0 for label in label_names}
    eval_pos = {label: 0 for label in label_names}

    def imbalance_after(assign_to_cal: bool, row: pd.Series) -> float:
        cal_size = len(cal_ids) + (1 if assign_to_cal else 0)
        eval_size = len(eval_ids) + (0 if assign_to_cal else 1)

        target_eval_size = len(merged) - target_cal_size
        score = abs(cal_size - target_cal_size) + abs(eval_size - target_eval_size)

        for label in label_names:
            y = int(row[f"{label}_true"] == 1)
            candidate_cal_pos = cal_pos[label] + (y if assign_to_cal else 0)
            candidate_eval_pos = eval_pos[label] + (0 if assign_to_cal else y)

            score += 3.0 * abs(candidate_cal_pos - target_pos[label])
            score += 3.0 * abs(candidate_eval_pos - (total_pos[label] - target_pos[label]))

        return float(score)

    for _, row in work.iterrows():
        if len(cal_ids) >= target_cal_size:
            choose_cal = False
        elif len(eval_ids) >= len(merged) - target_cal_size:
            choose_cal = True
        else:
            choose_cal = imbalance_after(True, row) <= imbalance_after(False, row)

        sample_id = str(row["sample_id"])
        if choose_cal:
            cal_ids.append(sample_id)
            for label in label_names:
                cal_pos[label] += int(row[f"{label}_true"] == 1)
        else:
            eval_ids.append(sample_id)
            for label in label_names:
                eval_pos[label] += int(row[f"{label}_true"] == 1)

    return pd.DataFrame(
        {
            "sample_id": cal_ids + eval_ids,
            "split": ["calibration"] * len(cal_ids) + ["evaluation"] * len(eval_ids),
        }
    )


def run_calibrated_threshold_evaluation(
    *,
    predictions_path: Path,
    labels_csv_path: Path,
    output_json_path: Path,
    output_calibration_best_csv_path: Path,
    output_evaluation_csv_path: Path,
    output_split_csv_path: Path,
    calibration_fraction: float = 0.5,
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

    split = _make_balanced_split(merged, label_names, calibration_fraction)
    merged = merged.merge(split, on="sample_id", how="inner")

    calibration = merged[merged["split"] == "calibration"].copy()
    evaluation = merged[merged["split"] == "evaluation"].copy()

    thresholds = _threshold_grid(threshold_start, threshold_stop, threshold_step)

    calibration_best_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    for label in label_names:
        y_true_cal = calibration[f"{label}_true"].astype(int)
        y_score_cal = calibration[f"{label}_score"].astype(float)

        calibration_rows: list[dict[str, Any]] = []

        for threshold in thresholds:
            metrics = _metrics_for_threshold(y_true_cal, y_score_cal, float(threshold))
            row = {
                "label": label,
                "selected_on": "calibration",
                "threshold": float(threshold),
                "n": int(len(y_true_cal)),
                "n_pos": int((y_true_cal == 1).sum()),
                "n_neg": int((y_true_cal == 0).sum()),
                **metrics,
            }
            calibration_rows.append(row)

        best = sorted(
            calibration_rows,
            key=lambda row: (
                row["f1"],
                row["balanced_accuracy"],
                row["specificity"],
                -abs(row["threshold"] - 0.5),
                -row["threshold"],
            ),
            reverse=True,
        )[0]
        calibration_best_rows.append(best)

        y_true_eval = evaluation[f"{label}_true"].astype(int)
        y_score_eval = evaluation[f"{label}_score"].astype(float)

        selected_eval = _metrics_for_threshold(y_true_eval, y_score_eval, best["threshold"])
        default_eval = _metrics_for_threshold(y_true_eval, y_score_eval, 0.5)

        evaluation_rows.append(
            {
                "label": label,
                "threshold_source": "calibration_selected",
                "threshold": best["threshold"],
                "n": int(len(y_true_eval)),
                "n_pos": int((y_true_eval == 1).sum()),
                "n_neg": int((y_true_eval == 0).sum()),
                **selected_eval,
            }
        )
        evaluation_rows.append(
            {
                "label": label,
                "threshold_source": "default_0.5",
                "threshold": 0.5,
                "n": int(len(y_true_eval)),
                "n_pos": int((y_true_eval == 1).sum()),
                "n_neg": int((y_true_eval == 0).sum()),
                **default_eval,
            }
        )

    calibration_best_df = pd.DataFrame(calibration_best_rows)
    evaluation_df = pd.DataFrame(evaluation_rows)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_calibration_best_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_evaluation_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_split_csv_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_best_df.to_csv(output_calibration_best_csv_path, index=False)
    evaluation_df.to_csv(output_evaluation_csv_path, index=False)
    split.to_csv(output_split_csv_path, index=False)

    selected_eval_df = evaluation_df[evaluation_df["threshold_source"] == "calibration_selected"]
    default_eval_df = evaluation_df[evaluation_df["threshold_source"] == "default_0.5"]

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
        "schema_version": "medshiftlab.calibrated_threshold_eval.v1",
        "source_predictions": predictions_path.name,
        "source_labels": labels_csv_path.name,
        "dataset_name": dataset_name,
        "model_name": predictions.get("model_name"),
        "model_version": predictions.get("model_version"),
        "preprocessing_version": predictions.get("preprocessing_version"),
        "normalization": normalization,
        "n_records": int(len(merged)),
        "split": {
            "method": "deterministic_greedy_multilabel_balanced",
            "calibration_fraction": calibration_fraction,
            "calibration_records": int(len(calibration)),
            "evaluation_records": int(len(evaluation)),
        },
        "threshold_grid": {
            "start": threshold_start,
            "stop": threshold_stop,
            "step": threshold_step,
        },
        "selection_rule": [
            "maximize_f1_on_calibration",
            "maximize_balanced_accuracy_on_calibration",
            "maximize_specificity_on_calibration",
            "minimize_distance_to_0.5",
            "prefer_lower_threshold",
        ],
        "calibration_best_thresholds": calibration_best_rows,
        "evaluation_metrics": evaluation_rows,
        "evaluation_aggregate": {
            "selected_mean_f1": float(selected_eval_df["f1"].mean()),
            "selected_mean_sensitivity": float(selected_eval_df["sensitivity"].mean()),
            "selected_mean_specificity": float(selected_eval_df["specificity"].mean()),
            "selected_mean_balanced_accuracy": float(selected_eval_df["balanced_accuracy"].mean()),
            "default_mean_f1": float(default_eval_df["f1"].mean()),
            "default_mean_sensitivity": float(default_eval_df["sensitivity"].mean()),
            "default_mean_specificity": float(default_eval_df["specificity"].mean()),
            "default_mean_balanced_accuracy": float(default_eval_df["balanced_accuracy"].mean()),
        },
        "notes": notes,
    }

    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-calibration-best-csv", required=True, type=Path)
    parser.add_argument("--output-evaluation-csv", required=True, type=Path)
    parser.add_argument("--output-split-csv", required=True, type=Path)
    parser.add_argument("--calibration-fraction", default=0.5, type=float)
    parser.add_argument("--threshold-start", default=0.0, type=float)
    parser.add_argument("--threshold-stop", default=1.0, type=float)
    parser.add_argument("--threshold-step", default=0.01, type=float)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_calibrated_threshold_evaluation(
        predictions_path=args.predictions,
        labels_csv_path=args.labels_csv,
        output_json_path=args.output_json,
        output_calibration_best_csv_path=args.output_calibration_best_csv,
        output_evaluation_csv_path=args.output_evaluation_csv,
        output_split_csv_path=args.output_split_csv,
        calibration_fraction=args.calibration_fraction,
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
                "split": payload["split"],
                "evaluation_aggregate": payload["evaluation_aggregate"],
                "outputs_written": {
                    "json": str(args.output_json),
                    "calibration_best_csv": str(args.output_calibration_best_csv),
                    "evaluation_csv": str(args.output_evaluation_csv),
                    "split_csv": str(args.output_split_csv),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
