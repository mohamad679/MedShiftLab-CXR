"""Tests for MedShiftLab-CXR evaluation metrics."""

from __future__ import annotations

import math

import pytest

from medshiftlab.evaluation import evaluate_binary_label, evaluate_label_metrics


def test_evaluate_binary_label_perfect_predictions() -> None:
    metrics = evaluate_binary_label(
        "Cardiomegaly",
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        threshold=0.5,
        n_bins=2,
    )

    assert metrics.label_name == "Cardiomegaly"
    assert metrics.n_available == 4
    assert metrics.n_binary == 4
    assert metrics.n_positive == 2
    assert metrics.n_negative == 2
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.ece == pytest.approx(0.15)
    assert metrics.f1 == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0


def test_evaluate_binary_label_uses_soft_targets_for_brier_and_ece() -> None:
    metrics = evaluate_binary_label(
        "Atelectasis",
        y_true=[0, 1, None, 0.5],
        y_score=[0.1, 0.9, 0.7, 0.6],
        threshold=0.5,
    )

    assert metrics.n_available == 3
    assert metrics.n_binary == 2
    assert metrics.n_positive == 1
    assert metrics.n_negative == 1
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
    assert metrics.brier_score == pytest.approx(0.01)
    assert metrics.ece is not None
    assert metrics.f1 == 1.0


def test_evaluate_binary_label_returns_none_for_single_class_discrimination() -> None:
    metrics = evaluate_binary_label(
        "Pleural Effusion",
        y_true=[0, 0, 0],
        y_score=[0.1, 0.2, 0.3],
    )

    assert metrics.n_binary == 3
    assert metrics.n_positive == 0
    assert metrics.n_negative == 3
    assert metrics.auroc is None
    assert metrics.auprc is None
    assert metrics.sensitivity is None
    assert metrics.specificity == 1.0


def test_evaluate_binary_label_ignores_missing_targets_and_scores() -> None:
    metrics = evaluate_binary_label(
        "Pneumonia",
        y_true=[0, 1, "", math.nan, None],
        y_score=[0.1, 0.8, 0.9, 0.2, None],
    )

    assert metrics.n_available == 2
    assert metrics.n_binary == 2
    assert metrics.auroc == 1.0


def test_evaluate_binary_label_ece_includes_score_one_in_last_bin() -> None:
    metrics = evaluate_binary_label(
        "Pneumonia",
        y_true=[0],
        y_score=[1.0],
        n_bins=10,
    )

    assert metrics.ece == 1.0


def test_evaluate_binary_label_ece_matches_mixed_calibration_case() -> None:
    metrics = evaluate_binary_label(
        "Pneumonia",
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.4, 0.8, 1.0],
        n_bins=2,
    )

    assert metrics.ece == pytest.approx(0.325)


def test_evaluate_binary_label_empty_available_values_have_no_ece() -> None:
    metrics = evaluate_binary_label(
        "Pneumonia",
        y_true=[None, ""],
        y_score=[0.2, None],
    )

    assert metrics.n_available == 0
    assert metrics.ece is None


def test_evaluate_binary_label_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="score must be between"):
        evaluate_binary_label(
            "Pneumothorax",
            y_true=[0, 1],
            y_score=[0.2, 1.5],
        )


def test_evaluate_binary_label_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_binary_label(
            "Pneumothorax",
            y_true=[0, 1],
            y_score=[0.2],
        )


def test_evaluate_label_metrics_evaluates_selected_labels() -> None:
    results = evaluate_label_metrics(
        y_true_by_label={
            "Atelectasis": [0, 1],
            "Cardiomegaly": [1, 0],
        },
        y_score_by_label={
            "Atelectasis": [0.1, 0.9],
            "Cardiomegaly": [0.8, 0.2],
        },
        labels=["Cardiomegaly"],
    )

    assert tuple(results) == ("Cardiomegaly",)
    assert results["Cardiomegaly"].auroc == 1.0


def test_evaluate_label_metrics_rejects_missing_scores() -> None:
    with pytest.raises(ValueError, match="Missing y_score"):
        evaluate_label_metrics(
            y_true_by_label={"Atelectasis": [0, 1]},
            y_score_by_label={},
        )
