#!/usr/bin/env python3
"""Build comparison tables and figures from four CheXpert metadata summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


STRATEGY_RUN_SUFFIXES = (
    ("U-ignore", "u_ignore"),
    ("U-zero", "u_zero"),
    ("U-one", "u_one"),
    ("U-soft", "u_soft"),
)
LABEL_SUMMARY_FILENAME = "chexpert_label_summary.csv"
DATASET_SUMMARY_FILENAME = "chexpert_dataset_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build CheXpert uncertainty-strategy comparison tables and figures "
            "from existing per-strategy metadata summaries."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results/real_runs",
        help="Root containing per-strategy result directories.",
    )
    parser.add_argument(
        "--figures-root",
        default="figures",
        help="Root where the comparison figure directory will be written.",
    )
    parser.add_argument(
        "--run-prefix",
        default="chexpert_train_chexbert",
        help="Prefix shared by the four per-strategy run directories.",
    )
    return parser


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required per-strategy input file: {path}"
        )
    return path


def _load_inputs(
    results_root: Path,
    run_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_frames: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []

    for strategy, suffix in STRATEGY_RUN_SUFFIXES:
        run_dir = results_root / f"{run_prefix}_{suffix}"
        label_path = _require_file(run_dir / LABEL_SUMMARY_FILENAME)
        dataset_path = _require_file(run_dir / DATASET_SUMMARY_FILENAME)

        label_frame = pd.read_csv(label_path)
        required_label_columns = {
            "label_name",
            "available_count",
            "missing_count",
            "positive_count",
            "negative_count",
            "soft_count",
            "positive_prevalence",
            "mean_target",
        }
        missing_columns = sorted(required_label_columns - set(label_frame.columns))
        if missing_columns:
            raise ValueError(
                f"Missing required label-summary columns in {label_path}: "
                + ", ".join(missing_columns)
            )
        label_frame.insert(0, "uncertainty_strategy", strategy)
        label_frames.append(label_frame)

        dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        required_dataset_fields = (
            "dataset_name",
            "n_records",
            "n_patients",
            "n_records_without_patient_id",
        )
        missing_fields = [
            field for field in required_dataset_fields if field not in dataset_payload
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required dataset-summary fields in {dataset_path}: "
                + ", ".join(missing_fields)
            )
        dataset_rows.append(
            {
                "uncertainty_strategy": strategy,
                **{field: dataset_payload[field] for field in required_dataset_fields},
            }
        )

    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(dataset_rows)


def _plot_grouped_metric(
    label_frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    strategies = [strategy for strategy, _ in STRATEGY_RUN_SUFFIXES]
    labels = list(dict.fromkeys(label_frame["label_name"]))
    positions = list(range(len(labels)))
    bar_width = 0.8 / len(strategies)

    figure, axis = plt.subplots(figsize=(11, 5))
    for strategy_index, strategy in enumerate(strategies):
        strategy_frame = label_frame[
            label_frame["uncertainty_strategy"] == strategy
        ].set_index("label_name")
        values = pd.to_numeric(
            strategy_frame.reindex(labels)[metric], errors="coerce"
        )
        offsets = [
            position - 0.4 + (strategy_index + 0.5) * bar_width
            for position in positions
        ]
        axis.bar(offsets, values, width=bar_width, label=strategy)

    axis.set_title(title)
    axis.set_xlabel("Label")
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    figures_root = Path(args.figures_root)

    label_frame, dataset_frame = _load_inputs(results_root, args.run_prefix)

    comparison_name = f"{args.run_prefix}_uncertainty_comparison"
    results_output_dir = results_root / comparison_name
    figures_output_dir = figures_root / comparison_name
    results_output_dir.mkdir(parents=True, exist_ok=True)
    figures_output_dir.mkdir(parents=True, exist_ok=True)

    label_output_path = (
        results_output_dir / "chexpert_uncertainty_strategy_label_summary.csv"
    )
    dataset_output_path = (
        results_output_dir / "chexpert_uncertainty_strategy_dataset_summary.csv"
    )
    label_frame.to_csv(label_output_path, index=False)
    dataset_frame.to_csv(dataset_output_path, index=False)
    print(label_output_path)
    print(dataset_output_path)

    _plot_grouped_metric(
        label_frame,
        metric="mean_target",
        title="Mean target by uncertainty strategy",
        ylabel="Mean target",
        output_path=figures_output_dir / "mean_target_by_uncertainty_strategy.png",
    )
    _plot_grouped_metric(
        label_frame,
        metric="positive_prevalence",
        title="Positive prevalence by uncertainty strategy",
        ylabel="Positive prevalence",
        output_path=(
            figures_output_dir / "positive_prevalence_by_uncertainty_strategy.png"
        ),
    )
    _plot_grouped_metric(
        label_frame,
        metric="soft_count",
        title="Soft counts by label",
        ylabel="Soft count",
        output_path=figures_output_dir / "soft_counts_by_label.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
