from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_vindr_inputs(tmp_path: Path) -> tuple[Path, Path]:
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
    annotations.to_csv(annotations_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return annotations_path, metadata_path


def _run_prepare_vindr_inputs(
    tmp_path: Path,
    *,
    image_root: Path | None = None,
    require_images: bool = False,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[1]
    annotations_path, metadata_path = _write_vindr_inputs(tmp_path)
    output_dir = tmp_path / "prepared"

    command = [
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
    ]
    if image_root is not None:
        command.extend(["--image-root", str(image_root)])
    if require_images:
        command.append("--require-images")
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_prepare_vindr_external_validation_inputs_writes_manifest_with_found_images(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    image_root = tmp_path / "images"
    image_root.mkdir()
    for filename in ["img001.jpg", "img002.png", "img003.dcm", "img004.jpeg"]:
        (image_root / filename).write_text("placeholder", encoding="utf-8")

    result = _run_prepare_vindr_inputs(tmp_path, image_root=image_root)

    assert result.returncode == 0
    assert "medshiftlab.vindr_external_validation_inputs.v1" in result.stdout

    labels_path = output_dir / "vindr_labels.csv"
    metadata_path = output_dir / "vindr_metadata.csv"
    manifest_path = output_dir / "vindr_manifest.csv"
    summary_path = output_dir / "vindr_prepare_summary.json"

    assert labels_path.exists()
    assert metadata_path.exists()
    assert manifest_path.exists()
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
    assert (
        img002[["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax"]]
        .sum()
        == 0.0
    )

    img003 = labels[labels["source_image_id"] == "img003"].iloc[0]
    assert img003["Pneumothorax"] == 1.0

    prepared_metadata = pd.read_csv(metadata_path)
    assert set(prepared_metadata.columns) == {
        "sample_id",
        "source_image_id",
        "sex",
        "view_position",
        "age_bucket",
    }

    manifest = pd.read_csv(manifest_path)
    assert manifest["image_found"].tolist() == [True, True, True, True]
    assert manifest["image_path"].tolist() == ["img001.jpg", "img002.png", "img003.dcm", "img004.jpeg"]
    assert all(not Path(image_path).is_absolute() for image_path in manifest["image_path"].tolist())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "medshiftlab.vindr_external_validation_inputs.v1"
    assert summary["claim_level"]["external_validation_completed"] is False
    assert summary["label_summary"]["n_unique_images"] == 4
    assert summary["label_summary"]["positive_counts"]["Cardiomegaly"] == 1
    assert summary["label_summary"]["positive_counts"]["Pleural Effusion"] == 1
    assert summary["label_summary"]["positive_counts"]["Pneumothorax"] == 1
    assert summary["label_summary"]["positive_counts"]["Atelectasis"] == 1
    assert "Other lesion" in summary["label_summary"]["unmapped_source_label_counts"]
    assert summary["manifest_summary"]["n_images_found"] == 4
    assert summary["manifest_summary"]["n_images_missing"] == 0


def test_prepare_vindr_external_validation_inputs_fails_when_image_root_has_no_matches(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "unrelated.jpg").write_text("placeholder", encoding="utf-8")

    result = _run_prepare_vindr_inputs(tmp_path, image_root=image_root)

    assert result.returncode != 0
    assert "No images were found under --image-root" in result.stdout
    assert not (output_dir / "vindr_prepare_summary.json").exists()
    assert not (output_dir / "vindr_manifest.csv").exists()


def test_prepare_vindr_external_validation_inputs_fails_when_image_root_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    image_root = tmp_path / "missing_images"

    result = _run_prepare_vindr_inputs(tmp_path, image_root=image_root)

    assert result.returncode != 0
    assert "--image-root does not exist" in result.stdout
    assert not (output_dir / "vindr_prepare_summary.json").exists()


def test_prepare_vindr_external_validation_inputs_require_images_fails_when_some_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    image_root = tmp_path / "images"
    image_root.mkdir()
    for filename in ["img001.jpg", "img003.png"]:
        (image_root / filename).write_text("placeholder", encoding="utf-8")

    result = _run_prepare_vindr_inputs(tmp_path, image_root=image_root, require_images=True)

    assert result.returncode != 0
    assert "--require-images was set" in result.stdout
    assert not (output_dir / "vindr_prepare_summary.json").exists()
