"""Tests for the MedShiftLab-CXR evaluation table interface."""

from __future__ import annotations

import pytest

from medshiftlab.evaluation import (
    EvaluationReport,
    build_label_matrices_from_rows,
    create_evaluation_report_from_rows,
)


def test_build_label_matrices_from_rows_returns_label_wise_values() -> None:
    rows = [
        {
            "image_id": "img001",
            "true_Atelectasis": 0,
            "score_Atelectasis": 0.2,
            "true_Cardiomegaly": 1,
            "score_Cardiomegaly": 0.8,
        },
        {
            "image_id": "img002",
            "true_Atelectasis": None,
            "score_Atelectasis": "nan",
            "true_Cardiomegaly": "",
            "score_Cardiomegaly": 0.3,
        },
    ]

    y_true, y_score = build_label_matrices_from_rows(
        rows, ["Atelectasis", "Cardiomegaly"]
    )

    assert y_true == {
        "Atelectasis": [0, None],
        "Cardiomegaly": [1, ""],
    }
    assert y_score == {
        "Atelectasis": [0.2, "nan"],
        "Cardiomegaly": [0.8, 0.3],
    }


def test_create_evaluation_report_from_rows_builds_metadata_and_metrics() -> None:
    rows = [
        {"true_Atelectasis": 0, "score_Atelectasis": 0.1},
        {"true_Atelectasis": 1, "score_Atelectasis": 0.9},
        {"true_Atelectasis": 0, "score_Atelectasis": 0.2},
        {"true_Atelectasis": 1, "score_Atelectasis": 0.8},
    ]

    report = create_evaluation_report_from_rows(
        rows=rows,
        labels=["Atelectasis"],
        dataset_name="CheXpert",
        model_name="mock-pretrained-cxr",
        split="validation",
        uncertainty_strategy="U-ignore",
        threshold=0.5,
        n_bins=2,
        notes="table adapter test",
    )

    assert isinstance(report, EvaluationReport)
    assert report.metadata.dataset_name == "CheXpert"
    assert report.metadata.model_name == "mock-pretrained-cxr"
    assert report.metadata.split == "validation"
    assert report.metadata.uncertainty_strategy == "U-ignore"
    assert report.metadata.notes == "table adapter test"
    assert report.label_metrics["Atelectasis"].auroc == 1.0
    assert report.label_metrics["Atelectasis"].auprc == 1.0
    assert report.aggregate_metrics.n_labels == 1


def test_build_label_matrices_from_rows_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="rows must not be empty"):
        build_label_matrices_from_rows([], ["Atelectasis"])


def test_build_label_matrices_from_rows_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        build_label_matrices_from_rows(
            [{"true_Atelectasis": 0, "score_Atelectasis": 0.2}], []
        )


def test_build_label_matrices_from_rows_rejects_missing_target_column() -> None:
    with pytest.raises(ValueError, match="true_Atelectasis"):
        build_label_matrices_from_rows(
            [{"score_Atelectasis": 0.2}], ["Atelectasis"]
        )


def test_build_label_matrices_from_rows_rejects_missing_score_column() -> None:
    with pytest.raises(ValueError, match="score_Atelectasis"):
        build_label_matrices_from_rows(
            [{"true_Atelectasis": 0}], ["Atelectasis"]
        )


def test_custom_target_and_score_prefixes_work() -> None:
    rows = [
        {"target_Atelectasis": 0, "probability_Atelectasis": 0.25},
        {"target_Atelectasis": 1, "probability_Atelectasis": 0.75},
    ]

    y_true, y_score = build_label_matrices_from_rows(
        rows,
        ["Atelectasis"],
        target_prefix="target_",
        score_prefix="probability_",
    )
    report = create_evaluation_report_from_rows(
        rows,
        ["Atelectasis"],
        dataset_name="custom-table",
        model_name="mock-model",
        target_prefix="target_",
        score_prefix="probability_",
    )

    assert y_true == {"Atelectasis": [0, 1]}
    assert y_score == {"Atelectasis": [0.25, 0.75]}
    assert report.label_metrics["Atelectasis"].auroc == 1.0
