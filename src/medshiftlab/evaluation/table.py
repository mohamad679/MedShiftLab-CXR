"""Row-based table adapters for MedShiftLab-CXR evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from medshiftlab.evaluation.report import EvaluationReport, create_evaluation_report


def build_label_matrices_from_rows(
    rows: Iterable[Mapping[str, Any]],
    labels: Iterable[str],
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> tuple[
    dict[str, list[float | int | None]],
    dict[str, list[float | int | None]],
]:
    """Transpose row-based targets and scores into label-wise mappings."""

    row_list = list(rows)
    label_list = list(labels)

    if not row_list:
        raise ValueError("rows must not be empty")
    if not label_list:
        raise ValueError("labels must not be empty")

    y_true_by_label: dict[str, list[float | int | None]] = {
        label: [] for label in label_list
    }
    y_score_by_label: dict[str, list[float | int | None]] = {
        label: [] for label in label_list
    }

    for row_index, row in enumerate(row_list):
        for label in label_list:
            target_column = f"{target_prefix}{label}"
            score_column = f"{score_prefix}{label}"

            if target_column not in row:
                raise ValueError(
                    f"Missing required target column {target_column!r} in row {row_index}"
                )
            if score_column not in row:
                raise ValueError(
                    f"Missing required score column {score_column!r} in row {row_index}"
                )

            y_true_by_label[label].append(row[target_column])
            y_score_by_label[label].append(row[score_column])

    return y_true_by_label, y_score_by_label


def create_evaluation_report_from_rows(
    rows: Iterable[Mapping[str, Any]],
    labels: Iterable[str],
    dataset_name: str,
    model_name: str,
    split: str | None = None,
    uncertainty_strategy: str | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    notes: str | None = None,
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> EvaluationReport:
    """Create an evaluation report from a row-based prediction table."""

    label_list = list(labels)
    y_true_by_label, y_score_by_label = build_label_matrices_from_rows(
        rows=rows,
        labels=label_list,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )

    return create_evaluation_report(
        dataset_name=dataset_name,
        model_name=model_name,
        y_true_by_label=y_true_by_label,
        y_score_by_label=y_score_by_label,
        labels=label_list,
        split=split,
        uncertainty_strategy=uncertainty_strategy,
        threshold=threshold,
        n_bins=n_bins,
        notes=notes,
    )
