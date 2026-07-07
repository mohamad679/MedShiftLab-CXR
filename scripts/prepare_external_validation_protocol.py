#!/usr/bin/env python3
"""Prepare bounded external-validation manifest and label-table scaffolding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from medshiftlab.data import (
    DEFAULT_EXTERNAL_PROTOCOL_LIMIT,
    MAX_SAFE_EXTERNAL_PROTOCOL_LIMIT_WITHOUT_OVERRIDE,
    prepare_external_validation_protocol,
    write_external_validation_label_table_csv,
    write_external_validation_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("mimic_cxr_jpg", "vindr_cxr"),
        help="Canonical external dataset name.",
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to a local external-dataset-like metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Local output directory for optional manifest and label-table files. "
            "Use an ignored private-results location."
        ),
    )
    parser.add_argument(
        "--protocol-config",
        help="Optional dataset-specific external protocol config YAML path.",
    )
    parser.add_argument(
        "--label-mapping-config",
        help="Optional dataset-specific label harmonization config YAML path.",
    )
    parser.add_argument(
        "--internal-manifest",
        help="Optional internal/dev manifest CSV for patient-overlap checks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_EXTERNAL_PROTOCOL_LIMIT,
        help=(
            "Maximum number of metadata rows to process. "
            f"Default: {DEFAULT_EXTERNAL_PROTOCOL_LIMIT}."
        ),
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=(
            "Allow metadata limits above the built-in safe threshold. "
            "Use only for explicit local/manual runs."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the validated external manifest CSV.",
    )
    parser.add_argument(
        "--write-label-table",
        action="store_true",
        help="Write the Phase 6-compatible external label-table CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        summary = run_preparation(args)
    except FileNotFoundError:
        print(
            "External validation setup error: required local file not found.",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(f"External validation setup error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_preparation(args: argparse.Namespace) -> dict[str, object]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if (
        args.limit > MAX_SAFE_EXTERNAL_PROTOCOL_LIMIT_WITHOUT_OVERRIDE
        and not args.allow_large_run
    ):
        raise ValueError(
            f"--limit {args.limit} exceeds the safe metadata-preparation threshold "
            f"of {MAX_SAFE_EXTERNAL_PROTOCOL_LIMIT_WITHOUT_OVERRIDE}. Re-run with "
            "--allow-large-run only for an explicit local/manual execution."
        )

    preparation = prepare_external_validation_protocol(
        args.dataset,
        args.metadata_csv,
        protocol_config_path=args.protocol_config,
        label_mapping_config_path=args.label_mapping_config,
        max_rows=args.limit,
        internal_manifest_path=args.internal_manifest,
    )

    written_outputs: dict[str, str] = {}
    if args.write_manifest or args.write_label_table:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.write_manifest:
            manifest_path = write_external_validation_manifest_csv(
                preparation.manifest_rows,
                output_dir / f"{args.dataset}_external_manifest.csv",
            )
            written_outputs["manifest"] = manifest_path.name

        if args.write_label_table:
            label_table_path = write_external_validation_label_table_csv(
                preparation.label_table_rows,
                output_dir / f"{args.dataset}_external_labels.csv",
                label_names=preparation.label_names,
            )
            written_outputs["label_table"] = label_table_path.name

    return {
        "dataset_name": args.dataset,
        "processed_records": len(preparation.manifest_rows),
        "limit": args.limit,
        "excluded_source_labels": list(preparation.excluded_source_labels),
        "label_names": list(preparation.label_names),
        "patient_overlap_check_requested": args.internal_manifest is not None,
        "patient_overlap_detected": bool(preparation.overlapping_internal_patient_ids),
        "outputs_written": written_outputs,
        "phase6_evaluation_input_compatible": True,
        "manual_only": True,
        "external_validation_completed": False,
        "benchmark_completed": False,
        "clinical_validation_completed": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
