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
