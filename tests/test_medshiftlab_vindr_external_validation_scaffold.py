from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_prepare_vindr_external_validation_inputs_writes_labels_metadata_and_summary(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]

    annotations = pd.DataFrame(
        {
            "image_id": [
                "img001",
                "img001",
                "img002",
                "img003",
                "img004",
                "img004",
            ],
            "class_name": [
                "Cardiomegaly",
                "Pleural effusion",
                "No finding",
                "Pneumothorax",
                "Atelectasis",
                "Other lesion",
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "image_id": ["img001", "img002", "img003", "img004"],
            "sex": ["Female", "Male", "Female", "Male"],
            "view_position": ["PA", "AP", "AP", "PA"],
            "age_bucket": ["40-59", "60-79", "<40", "80+"],
        }
    )

    annotations_path = tmp_path / "annotations.csv"
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "prepared"

    annotations.to_csv(annotations_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/prepare_vindr_external_validation_inputs.py"),
            "--annotations-csv",
            str(annotations_path),
            "--mapping-json",
            str(repo / "configs/evaluation/vindr_cxr_label_mapping.json"),
            "--metadata-csv",
            str(metadata_path),
            "--metadata-columns",
            "sex",
            "view_position",
            "age_bucket",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "medshiftlab.vindr_external_validation_inputs.v1" in result.stdout

    labels_path = output_dir / "vindr_labels.csv"
    prepared_metadata_path = output_dir / "vindr_metadata.csv"
    summary_path = output_dir / "vindr_prepare_summary.json"

    assert labels_path.exists()
    assert prepared_metadata_path.exists()
    assert summary_path.exists()

    labels = pd.read_csv(labels_path)
    assert list(labels.columns) == [
        "sample_id",
        "source_image_id",
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
    ]

    img001 = labels[labels["source_image_id"] == "img001"].iloc[0]
    assert img001["Cardiomegaly"] == 1.0
    assert img001["Pleural Effusion"] == 1.0

    img002 = labels[labels["source_image_id"] == "img002"].iloc[0]
    assert img002[["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax"]].sum() == 0.0

    img003 = labels[labels["source_image_id"] == "img003"].iloc[0]
    assert img003["Pneumothorax"] == 1.0

    prepared_metadata = pd.read_csv(prepared_metadata_path)
    assert set(prepared_metadata.columns) == {
        "sample_id",
        "source_image_id",
        "sex",
        "view_position",
        "age_bucket",
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "medshiftlab.vindr_external_validation_inputs.v1"
    assert summary["claim_level"]["external_validation_completed"] is False
    assert summary["label_summary"]["n_unique_images"] == 4
    assert summary["label_summary"]["positive_counts"]["Cardiomegaly"] == 1
    assert summary["label_summary"]["positive_counts"]["Pleural Effusion"] == 1
    assert summary["label_summary"]["positive_counts"]["Pneumothorax"] == 1
    assert summary["label_summary"]["positive_counts"]["Atelectasis"] == 1
    assert "Other lesion" in summary["label_summary"]["unmapped_source_label_counts"]
