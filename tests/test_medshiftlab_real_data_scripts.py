"""Tests for the real-data preparation scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_tiny_chexpert_csv(csv_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "Sex": "Female",
                "Age": 65,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "PA",
                "Atelectasis": -1,
                "Cardiomegaly": 1,
                "Pleural Effusion": 0,
                "Pneumonia": "",
                "Pneumothorax": None,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                "Sex": "Male",
                "Age": 72,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "AP",
                "Atelectasis": 0,
                "Cardiomegaly": -1,
                "Pleural Effusion": 1,
                "Pneumonia": 0,
                "Pneumothorax": "",
                "No Finding": 0,
            },
        ]
    ).to_csv(csv_path, index=False)


def _write_strategy_summaries(results_root: Path, run_prefix: str) -> None:
    strategies = (
        ("u_ignore", 0),
        ("u_zero", 0),
        ("u_one", 0),
        ("u_soft", 4),
    )
    for suffix, soft_count in strategies:
        run_dir = results_root / f"{run_prefix}_{suffix}"
        run_dir.mkdir(parents=True)
        (run_dir / "chexpert_dataset_summary.json").write_text(
            json.dumps(
                {
                    "dataset_name": "CheXpert",
                    "n_records": 10,
                    "n_patients": 5,
                    "n_records_without_patient_id": 0,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "label_name": "Atelectasis",
                    "available_count": 10,
                    "missing_count": 0,
                    "positive_count": 4,
                    "negative_count": 6 - soft_count,
                    "soft_count": soft_count,
                    "positive_prevalence": 0.4,
                    "mean_target": 0.6 if soft_count else 0.4,
                },
                {
                    "label_name": "Pneumonia",
                    "available_count": 8,
                    "missing_count": 2,
                    "positive_count": 2,
                    "negative_count": 6,
                    "soft_count": 0,
                    "positive_prevalence": 0.25,
                    "mean_target": 0.25,
                },
            ]
        ).to_csv(run_dir / "chexpert_label_summary.csv", index=False)


def test_real_data_scripts_write_summary_and_plots(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "chexpert_tiny.csv"
    summary_output_dir = tmp_path / "results" / "real_runs"
    figure_output_dir = tmp_path / "figures"
    _write_tiny_chexpert_csv(csv_path)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."

    summary_result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_chexpert_metadata.py",
            "--csv-path",
            str(csv_path),
            "--uncertainty-strategy",
            "U-soft",
            "--output-dir",
            str(summary_output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    summary_json_path = summary_output_dir / "chexpert_dataset_summary.json"
    label_csv_path = summary_output_dir / "chexpert_label_summary.csv"

    assert str(summary_json_path) in summary_result.stdout
    assert str(label_csv_path) in summary_result.stdout
    assert summary_json_path.exists()
    assert label_csv_path.exists()

    summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["dataset_name"] == "CheXpert"
    assert summary_payload["n_records"] == 2

    label_frame = pd.read_csv(label_csv_path)
    assert set(label_frame["label_name"]) == {
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
        "No Finding",
    }

    plot_result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_dataset_summary.py",
            "--label-summary-csv",
            str(label_csv_path),
            "--output-dir",
            str(figure_output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    for figure_name in (
        "label_available_counts.png",
        "label_positive_prevalence.png",
        "label_mean_target.png",
    ):
        figure_path = figure_output_dir / figure_name
        assert str(figure_path) in plot_result.stdout
        assert figure_path.exists()


def test_build_uncertainty_comparison_script_writes_tables_and_figures(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results" / "real_runs"
    figures_root = tmp_path / "figures"
    run_prefix = "toy_chexpert"
    _write_strategy_summaries(results_root, run_prefix)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_chexpert_uncertainty_comparison.py",
            "--results-root",
            str(results_root),
            "--figures-root",
            str(figures_root),
            "--run-prefix",
            run_prefix,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    comparison_name = f"{run_prefix}_uncertainty_comparison"
    results_output_dir = results_root / comparison_name
    figures_output_dir = figures_root / comparison_name
    label_output = (
        results_output_dir / "chexpert_uncertainty_strategy_label_summary.csv"
    )
    dataset_output = (
        results_output_dir / "chexpert_uncertainty_strategy_dataset_summary.csv"
    )

    assert label_output.exists()
    assert dataset_output.exists()
    assert str(label_output) in result.stdout
    assert str(dataset_output) in result.stdout
    assert set(pd.read_csv(label_output)["uncertainty_strategy"]) == {
        "U-ignore",
        "U-zero",
        "U-one",
        "U-soft",
    }

    for figure_name in (
        "mean_target_by_uncertainty_strategy.png",
        "positive_prevalence_by_uncertainty_strategy.png",
        "soft_counts_by_label.png",
    ):
        figure_path = figures_output_dir / figure_name
        assert figure_path.exists()
        assert str(figure_path) in result.stdout


def test_build_uncertainty_comparison_script_fails_for_missing_inputs(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_chexpert_uncertainty_comparison.py",
            "--results-root",
            str(tmp_path / "results"),
            "--figures-root",
            str(tmp_path / "figures"),
            "--run-prefix",
            "missing",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing required per-strategy input file" in result.stderr
