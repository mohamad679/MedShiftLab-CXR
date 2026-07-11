#!/usr/bin/env python3
"""Run aggregate cross-dataset bootstrap comparison from prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from medshiftlab.experiments import run_cross_dataset_bootstrap_from_files
from medshiftlab.reporting import write_cross_dataset_bootstrap_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--reference-labels-csv", required=True)
    parser.add_argument("--external-predictions", required=True)
    parser.add_argument("--external-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--metrics",
        default="auroc,auprc,brier_score,ece",
        help="Comma-separated metrics: auroc, auprc, brier_score, ece.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())

    try:
        report = run_cross_dataset_bootstrap_from_files(
            reference_predictions_path=args.reference_predictions,
            reference_labels_csv_path=args.reference_labels_csv,
            external_predictions_path=args.external_predictions,
            external_labels_csv_path=args.external_labels_csv,
            iterations=args.iterations,
            seed=args.seed,
            confidence_level=args.confidence_level,
            n_bins=args.n_bins,
            metrics=metrics,
        )
        outputs = write_cross_dataset_bootstrap_bundle(report, args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cross-dataset bootstrap error: {error}", file=sys.stderr)
        return 2

    summary = {
        "schema_version": report.schema_version,
        "reference_dataset_name": report.reference_dataset_name,
        "external_dataset_name": report.external_dataset_name,
        "model_name": report.model_name,
        "shared_labels": report.shared_labels,
        "requested_metrics": report.requested_metrics,
        "iterations": report.iterations,
        "reference_resampling_unit": report.reference_resampling_unit,
        "external_resampling_unit": report.external_resampling_unit,
        "result_count": len(report.results),
        "outputs_written": {key: path.name for key, path in outputs.items()},
        "delta_definition": report.delta_definition,
        "manual_local_only": report.manual_local_only,
        "real_inference_performed": report.real_inference_performed,
        "clinical_validation_completed": report.clinical_validation_completed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
