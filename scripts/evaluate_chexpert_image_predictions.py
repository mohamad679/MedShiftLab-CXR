#!/usr/bin/env python3
"""Evaluate CheXpert targets against TorchXRayVision prediction columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from medshiftlab.evaluation.metrics import evaluate_binary_label

LABEL_TO_PREDICTION = {
    "Atelectasis": "pred_Atelectasis",
    "Cardiomegaly": "pred_Cardiomegaly",
    "Pleural Effusion": "pred_Effusion",
    "Pneumonia": "pred_Pneumonia",
    "Pneumothorax": "pred_Pneumothorax",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser


def evaluate_predictions(frame: pd.DataFrame, threshold: float, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, prediction_column in LABEL_TO_PREDICTION.items():
        missing = [column for column in (label, prediction_column) if column not in frame]
        if missing:
            raise ValueError("Missing prediction-table columns: " + ", ".join(missing))
        raw_targets = pd.to_numeric(frame[label], errors="coerce")
        soft_targets = raw_targets.replace(-1.0, 0.5)
        predictions = pd.to_numeric(frame[prediction_column], errors="coerce")
        metrics = evaluate_binary_label(
            label, soft_targets.tolist(), predictions.tolist(), threshold=threshold, n_bins=n_bins
        )
        available = soft_targets.notna() & predictions.notna()
        rows.append(
            {
                "label": label,
                "n_available_soft": metrics.n_available,
                "n_available_binary": metrics.n_binary,
                "n_positive_binary": metrics.n_positive,
                "n_negative_binary": metrics.n_negative,
                "mean_target_soft": float(soft_targets[available].mean()),
                "mean_prediction": float(predictions[available].mean()),
                "brier_soft": metrics.brier_score,
                "ece_soft": metrics.ece,
                "auroc_binary": metrics.auroc,
                "auprc_binary": metrics.auprc,
                "f1_binary_at_0_5": metrics.f1,
                "sensitivity_at_0_5": metrics.sensitivity,
                "specificity_at_0_5": metrics.specificity,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(pd.read_csv(args.predictions_csv), args.threshold, args.n_bins)
    csv_path = output_dir / "evaluation_label_metrics.csv"
    report_path = output_dir / "evaluation_report.json"
    metrics.to_csv(csv_path, index=False)
    report = {
        "evaluation_policy": {
            "uncertain_label_soft_value": 0.5,
            "binary_metrics_targets": [0, 1],
            "missing_labels": "omitted",
            "threshold": args.threshold,
            "threshold_tuning": False,
            "ece_bins": args.n_bins,
        },
        "labels": metrics.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(csv_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
