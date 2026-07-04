#!/usr/bin/env python3
"""Summarize CheXpert metadata CSV contents without loading images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from medshiftlab.data import load_chexpert_metadata_csv, summarize_chexpert_records
from medshiftlab.labels import load_default_label_ontology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a CheXpert metadata CSV without loading raw images."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the CheXpert metadata CSV.")
    parser.add_argument(
        "--uncertainty-strategy",
        required=True,
        choices=("U-ignore", "U-zero", "U-one", "U-soft"),
        help="CheXpert uncertainty handling strategy.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary JSON and CSV files will be written.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for small audits or smoke tests.",
    )
    return parser


def write_label_summary_csv(summary, output_path: Path) -> None:
    fieldnames = [
        "label_name",
        "available_count",
        "missing_count",
        "positive_count",
        "negative_count",
        "soft_count",
        "positive_prevalence",
        "mean_target",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label_name, label_summary in summary.labels.items():
            writer.writerow(
                {
                    "label_name": label_name,
                    "available_count": label_summary.available_count,
                    "missing_count": label_summary.missing_count,
                    "positive_count": label_summary.positive_count,
                    "negative_count": label_summary.negative_count,
                    "soft_count": label_summary.soft_count,
                    "positive_prevalence": label_summary.positive_prevalence,
                    "mean_target": label_summary.mean_target,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ontology = load_default_label_ontology()
    records = load_chexpert_metadata_csv(
        args.csv_path,
        ontology,
        args.uncertainty_strategy,
        max_rows=args.max_rows,
    )
    summary = summarize_chexpert_records(records)

    summary_json_path = output_dir / "chexpert_dataset_summary.json"
    label_csv_path = output_dir / "chexpert_label_summary.csv"

    summary_json_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    write_label_summary_csv(summary, label_csv_path)

    print(summary_json_path)
    print(label_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
