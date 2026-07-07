#!/usr/bin/env python3
"""Evaluate standardized prediction batches against a strict local label CSV."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from medshiftlab.experiments import (
    DEFAULT_EVALUATION_LIMIT,
    PredictionEvaluationConfig,
    run_prediction_batch_evaluation_from_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a standardized prediction batch JSON or CSV file.",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        help="CSV with sample_id/image_id and one column per evaluated label.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output path for the standardized evaluation report JSON.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional output path for the standardized evaluation label-metrics CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        help="Reserved for future calibration/figure export support.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="Reserved for future bootstrap CI support. Keep at 0 for now.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Scalar threshold used for F1, sensitivity, and specificity.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of calibration bins for ECE.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_EVALUATION_LIMIT,
        help=(
            "Maximum number of prediction records to evaluate. "
            f"Default: {DEFAULT_EVALUATION_LIMIT}."
        ),
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Allow evaluation limits above the built-in safe threshold.",
    )
    parser.add_argument("--split")
    parser.add_argument("--uncertainty-strategy")
    parser.add_argument("--notes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result, written_outputs = run_prediction_batch_evaluation_from_files(
            predictions_path=args.predictions,
            labels_csv_path=args.labels_csv,
            config=PredictionEvaluationConfig(
                threshold=args.threshold,
                n_bins=args.n_bins,
                limit=args.limit,
                allow_large_run=args.allow_large_run,
                split=args.split,
                uncertainty_strategy=args.uncertainty_strategy,
                notes=args.notes,
                bootstrap_iters=args.bootstrap_iters,
                figures_dir=args.figures_dir,
            ),
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        print(f"Real prediction evaluation error: {error}", file=sys.stderr)
        return 2

    summary = {
        "prediction_format": result.prediction_format,
        "dataset_name": result.report.metadata.dataset_name,
        "model_name": result.report.metadata.model_name,
        "evaluated_records": result.accounting.evaluated_records,
        "skipped_records": result.accounting.skipped_records,
        "threshold": result.report.metadata.threshold,
        "n_bins": result.report.metadata.n_bins,
        "metrics": [
            "auroc",
            "auprc",
            "f1",
            "sensitivity",
            "specificity",
            "brier_score",
            "ece",
            "confusion_counts",
        ],
        "support_status": result.support_status,
        "outputs_written": {
            key: path.name for key, path in written_outputs.items()
        },
        "manual_only": True,
        "full_benchmark_completed": False,
        "external_validation_completed": False,
        "clinical_validation_completed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
