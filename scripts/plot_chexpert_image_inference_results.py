#!/usr/bin/env python3
"""Generate metric-only figures from CheXpert image-inference results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {
    "label",
    "auroc_binary",
    "auprc_binary",
    "brier_soft",
    "ece_soft",
    "sensitivity_at_0_5",
    "specificity_at_0_5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _single_metric(frame: pd.DataFrame, column: str, title: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(frame["label"], frame[column])
    axis.set(title=title, xlabel="Label", ylabel=ylabel, ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _paired(frame: pd.DataFrame, columns: tuple[str, str], labels: tuple[str, str], title: str, path: Path) -> None:
    positions = list(range(len(frame)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar([x - width / 2 for x in positions], frame[columns[0]], width, label=labels[0])
    axis.bar([x + width / 2 for x in positions], frame[columns[1]], width, label=labels[1])
    axis.set(title=title, xlabel="Label", ylabel="Metric value", ylim=(0, 1))
    axis.set_xticks(positions, frame["label"], rotation=30)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_metrics(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("Missing metrics columns: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "auroc_by_label.png",
        output_dir / "auprc_by_label.png",
        output_dir / "calibration_metrics.png",
        output_dir / "sensitivity_specificity_at_0_5.png",
    ]
    _single_metric(frame, "auroc_binary", "Binary AUROC by label", "AUROC", outputs[0])
    _single_metric(frame, "auprc_binary", "Binary AUPRC by label", "AUPRC", outputs[1])
    _paired(frame, ("brier_soft", "ece_soft"), ("Brier", "ECE"), "Soft-label calibration metrics", outputs[2])
    _paired(
        frame,
        ("sensitivity_at_0_5", "specificity_at_0_5"),
        ("Sensitivity", "Specificity"),
        "Sensitivity and specificity at threshold 0.5",
        outputs[3],
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in plot_metrics(pd.read_csv(args.metrics_csv), Path(args.output_dir)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
