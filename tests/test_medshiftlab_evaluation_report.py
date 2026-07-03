"""Tests for MedShiftLab-CXR evaluation reports."""

from __future__ import annotations

import pytest

from medshiftlab.evaluation import (
    BinaryLabelMetrics,
    create_evaluation_report,
    summarize_evaluation_metrics,
)


def test_create_evaluation_report_wraps_label_metrics_and_metadata() -> None:
    report = create_evaluation_report(
        dataset_name="CheXpert",
        model_name="torchxrayvision-densenet121",
        split="validation",
        uncertainty_strategy="U-soft",
        threshold=0.5,
        n_bins=2,
        y_true_by_label={
            "Atelectasis": [0, 1, 0, 1],
            "Cardiomegaly": [0, 1, 1, 0],
        },
        y_score_by_label={
            "Atelectasis": [0.1, 0.9, 0.2, 0.8],
            "Cardiomegaly": [0.2, 0.8, 0.7, 0.3],
        },
    )

    assert report.metadata.dataset_name == "CheXpert"
    assert report.metadata.model_name == "torchxrayvision-densenet121"
    assert report.metadata.split == "validation"
    assert report.metadata.uncertainty_strategy == "U-soft"
    assert tuple(report.label_metrics) == ("Atelectasis", "Cardiomegaly")
    assert report.label_metrics["Atelectasis"].auroc == 1.0
    assert report.label_metrics["Cardiomegaly"].auroc == 1.0
    assert report.aggregate_metrics.n_labels == 2
    assert report.aggregate_metrics.mean_auroc == 1.0
    assert report.aggregate_metrics.mean_auprc == 1.0


def test_evaluation_report_to_flat_rows_exports_one_row_per_label() -> None:
    report = create_evaluation_report(
        dataset_name="VinDr-CXR",
        model_name="mock-pretrained-cxr",
        split="test",
        threshold=0.4,
        n_bins=5,
        y_true_by_label={
            "Pneumothorax": [0, 1],
            "Pleural Effusion": [1, 0],
        },
        y_score_by_label={
            "Pneumothorax": [0.2, 0.9],
            "Pleural Effusion": [0.8, 0.1],
        },
    )

    rows = report.to_flat_rows()

    assert len(rows) == 2
    assert rows[0]["dataset_name"] == "VinDr-CXR"
    assert rows[0]["model_name"] == "mock-pretrained-cxr"
    assert rows[0]["split"] == "test"
    assert rows[0]["threshold"] == 0.4
    assert rows[0]["n_bins"] == 5
    assert rows[0]["label_name"] == "Pneumothorax"
    assert rows[0]["auroc"] == 1.0
    assert rows[1]["label_name"] == "Pleural Effusion"


def test_summarize_evaluation_metrics_ignores_unavailable_values() -> None:
    metrics = {
        "A": BinaryLabelMetrics(
            label_name="A",
            n_available=2,
            n_binary=2,
            n_positive=1,
            n_negative=1,
            threshold=0.5,
            auroc=1.0,
            auprc=0.8,
            brier_score=0.1,
            ece=0.2,
            f1=1.0,
            sensitivity=1.0,
            specificity=1.0,
        ),
        "B": BinaryLabelMetrics(
            label_name="B",
            n_available=2,
            n_binary=2,
            n_positive=0,
            n_negative=2,
            threshold=0.5,
            auroc=None,
            auprc=None,
            brier_score=0.3,
            ece=0.4,
            f1=None,
            sensitivity=None,
            specificity=1.0,
        ),
    }

    aggregate = summarize_evaluation_metrics(metrics)

    assert aggregate.n_labels == 2
    assert aggregate.mean_auroc == 1.0
    assert aggregate.mean_auprc == 0.8
    assert aggregate.mean_brier_score == pytest.approx(0.2)
    assert aggregate.mean_ece == pytest.approx(0.3)
    assert aggregate.mean_f1 == 1.0
    assert aggregate.mean_sensitivity == 1.0
    assert aggregate.mean_specificity == 1.0


def test_create_evaluation_report_can_select_label_subset() -> None:
    report = create_evaluation_report(
        dataset_name="CheXpert",
        model_name="mock-model",
        labels=["Cardiomegaly"],
        y_true_by_label={
            "Atelectasis": [0, 1],
            "Cardiomegaly": [1, 0],
        },
        y_score_by_label={
            "Atelectasis": [0.1, 0.9],
            "Cardiomegaly": [0.8, 0.2],
        },
    )

    assert tuple(report.label_metrics) == ("Cardiomegaly",)
    assert report.aggregate_metrics.n_labels == 1
    assert report.aggregate_metrics.mean_auroc == 1.0
