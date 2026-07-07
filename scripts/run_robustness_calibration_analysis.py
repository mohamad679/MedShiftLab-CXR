#!/usr/bin/env python3
"""Run bounded robustness/calibration analysis on local prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from medshiftlab.experiments import (
    DEFAULT_EVALUATION_LIMIT,
    RobustnessAnalysisConfig,
    run_robustness_analysis_from_files,
)
from medshiftlab.reporting import write_robustness_report_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--metadata-csv",
        help="Optional sample_id-keyed CSV containing subgroup metadata.",
    )
    parser.add_argument(
        "--baseline-evaluation-json",
        help="Optional standardized evaluation report used for degradation flags.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional local directory for aggregate JSON/CSV reports.",
    )
    parser.add_argument(
        "--export-calibration-csv",
        action="store_true",
        help="Export ECE-compatible calibration-bin table data under --output-dir.",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument(
        "--bootstrap-metrics",
        default="auroc,brier_score,ece",
        help="Comma-separated supported scalar metrics.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--subgroup-columns",
        nargs="+",
        default=["dataset_name", "split", "uncertainty_strategy"],
    )
    parser.add_argument("--minimum-subgroup-size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--split")
    parser.add_argument("--uncertainty-strategy")
    parser.add_argument("--degradation-metric", default="auroc")
    parser.add_argument("--degradation-threshold", type=float)
    parser.add_argument("--maximum-ece", type=float)
    parser.add_argument("--maximum-brier-score", type=float)
    parser.add_argument("--limit", type=int, default=DEFAULT_EVALUATION_LIMIT)
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Explicitly allow a limit above the built-in manual-run guard.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = tuple(
        item.strip() for item in args.bootstrap_metrics.split(",") if item.strip()
    )
    try:
        report = run_robustness_analysis_from_files(
            predictions_path=args.predictions,
            labels_csv_path=args.labels_csv,
            metadata_csv_path=args.metadata_csv,
            baseline_evaluation_json=args.baseline_evaluation_json,
            config=RobustnessAnalysisConfig(
                threshold=args.threshold,
                n_bins=args.n_bins,
                limit=args.limit,
                allow_large_run=args.allow_large_run,
                split=args.split,
                uncertainty_strategy=args.uncertainty_strategy,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_metrics=metrics,
                seed=args.seed,
                confidence_level=args.confidence_level,
                subgroup_columns=tuple(args.subgroup_columns),
                minimum_subgroup_size=args.minimum_subgroup_size,
                degradation_metric=args.degradation_metric,
                degradation_threshold=args.degradation_threshold,
                maximum_ece=args.maximum_ece,
                maximum_brier_score=args.maximum_brier_score,
            ),
        )
        outputs = (
            write_robustness_report_bundle(
                report,
                args.output_dir,
                export_calibration_csv=args.export_calibration_csv,
            )
            if args.output_dir
            else {}
        )
        if args.export_calibration_csv and not args.output_dir:
            raise ValueError("--export-calibration-csv requires --output-dir")
    except (FileNotFoundError, ValueError) as error:
        print(f"Robustness analysis error: {error}", file=sys.stderr)
        return 2

    summary = {
        "schema_version": report.schema_version,
        "dataset_name": report.dataset_name,
        "model_name": report.model_name,
        "evaluated_records": report.evaluated_records,
        "calibration_labels": len(report.calibration),
        "bootstrap_intervals": len(report.bootstrap_intervals),
        "subgroup_metric_results": len(report.subgroup_analysis.results),
        "skipped_subgroup_labels": len(report.subgroup_analysis.skipped),
        "failure_flags": {
            "metric_degradations": len(report.failure_cases.metric_degradations),
            "poor_calibration": len(report.failure_cases.poor_calibration),
            "coverage_issues": len(report.failure_cases.coverage_issues),
        },
        "outputs_written": {key: path.name for key, path in outputs.items()},
        "support_status": report.support_status,
        "manual_local_only": True,
        "real_inference_performed": False,
        "full_benchmark_completed": False,
        "external_validation_completed": False,
        "clinical_validation_completed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
