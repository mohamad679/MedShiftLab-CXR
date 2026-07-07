#!/usr/bin/env python3
"""Prepare bounded CheXpert internal-protocol split and label-table scaffolding."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

from medshiftlab.data import (
    DEFAULT_CHEXPERT_PROTOCOL_LIMIT,
    DEFAULT_CHEXPERT_PROTOCOL_SEED,
    MAX_SAFE_CHEXPERT_PROTOCOL_LIMIT_WITHOUT_OVERRIDE,
    CheXpertSplitConfig,
    prepare_chexpert_internal_protocol,
    write_chexpert_label_table_csv,
    write_chexpert_split_manifest_csv,
)
from medshiftlab.labels import UncertaintyStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to a local CheXpert-style metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Local output directory for optional split-manifest and label-table files. "
            "Use an ignored private-results location."
        ),
    )
    parser.add_argument(
        "--uncertainty-strategy",
        default="all",
        choices=("all", "U-ignore", "U-zero", "U-one", "U-soft"),
        help="Single strategy to materialize, or 'all' for every supported strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CHEXPERT_PROTOCOL_SEED,
        help="Deterministic seed for patient-level split assignment.",
    )
    parser.add_argument(
        "--split-names",
        default="train,validation,test",
        help="Comma-separated split names, for example train,validation,test.",
    )
    parser.add_argument(
        "--split-fractions",
        default="0.70,0.15,0.15",
        help="Comma-separated split fractions that sum to 1.0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CHEXPERT_PROTOCOL_LIMIT,
        help=(
            "Maximum number of metadata rows to process. "
            f"Default: {DEFAULT_CHEXPERT_PROTOCOL_LIMIT}."
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
        "--write-split-manifest",
        action="store_true",
        help="Write the patient-level split manifest CSV.",
    )
    parser.add_argument(
        "--write-label-tables",
        action="store_true",
        help="Write one Phase 6-compatible label-table CSV per selected strategy.",
    )
    parser.add_argument(
        "--soft-value",
        type=float,
        default=0.5,
        help="Target value used for U-soft uncertain labels.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        summary = run_preparation(args)
    except FileNotFoundError:
        print(
            "CheXpert internal protocol error: metadata CSV not found.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, RuntimeError) as error:
        print(f"CheXpert internal protocol error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_preparation(args: argparse.Namespace) -> dict[str, object]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if (
        args.limit > MAX_SAFE_CHEXPERT_PROTOCOL_LIMIT_WITHOUT_OVERRIDE
        and not args.allow_large_run
    ):
        raise ValueError(
            f"--limit {args.limit} exceeds the safe metadata-preparation threshold "
            f"of {MAX_SAFE_CHEXPERT_PROTOCOL_LIMIT_WITHOUT_OVERRIDE}. Re-run with "
            "--allow-large-run only for an explicit local/manual execution."
        )

    split_config = CheXpertSplitConfig(
        split_names=_parse_split_names(args.split_names),
        split_fractions=_parse_split_fractions(args.split_fractions),
        seed=args.seed,
    )
    strategies = _parse_strategies(args.uncertainty_strategy)
    preparation = prepare_chexpert_internal_protocol(
        args.metadata_csv,
        split_config=split_config,
        uncertainty_strategies=strategies,
        max_rows=args.limit,
        soft_value=args.soft_value,
    )

    written_outputs: dict[str, object] = {}
    if args.write_split_manifest or args.write_label_tables:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.write_split_manifest:
            split_manifest_path = write_chexpert_split_manifest_csv(
                preparation.split_manifest_rows,
                output_dir / "chexpert_internal_split_manifest.csv",
            )
            written_outputs["split_manifest"] = split_manifest_path.name

        if args.write_label_tables:
            written_outputs["label_tables"] = {
                strategy: write_chexpert_label_table_csv(
                    rows,
                    output_dir / f"chexpert_labels_{_strategy_slug(strategy)}.csv",
                    label_names=preparation.label_names,
                ).name
                for strategy, rows in preparation.label_table_rows_by_strategy.items()
            }

    split_counts: dict[str, int] = {}
    for row in preparation.split_manifest_rows:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1

    return {
        "dataset_name": "chexpert",
        "processed_records": len(preparation.split_manifest_rows),
        "limit": args.limit,
        "split_level": "patient",
        "split_names": list(preparation.split_config.split_names),
        "split_counts": split_counts,
        "uncertainty_strategies": list(preparation.label_table_rows_by_strategy),
        "label_names": list(preparation.label_names),
        "manual_only": True,
        "phase6_evaluation_input_compatible": True,
        "outputs_written": written_outputs,
        "full_chexpert_inference_completed": False,
        "internal_benchmark_completed": False,
        "external_validation_completed": False,
        "clinical_validation_completed": False,
    }


def _parse_split_names(raw_value: str) -> tuple[str, ...]:
    split_names = tuple(
        item.strip() for item in raw_value.split(",") if item.strip()
    )
    if not split_names:
        raise ValueError("--split-names must contain at least one non-empty value")
    return split_names


def _parse_split_fractions(raw_value: str) -> tuple[float, ...]:
    pieces = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not pieces:
        raise ValueError("--split-fractions must contain at least one value")

    try:
        return tuple(float(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("--split-fractions must contain only numeric values") from error


def _parse_strategies(raw_value: str) -> tuple[UncertaintyStrategy, ...]:
    if raw_value == "all":
        return tuple(strategy for strategy in UncertaintyStrategy)
    return (UncertaintyStrategy(raw_value),)


def _strategy_slug(strategy: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", strategy.lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
