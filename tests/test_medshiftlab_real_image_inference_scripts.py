"""Tests for the reproducible real-image inference scripts using synthetic data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment(*extra_pythonpath: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in extra_pythonpath), "src", "."])
    return environment


def _png_bytes(size: tuple[int, int], value: int = 64) -> bytes:
    buffer = BytesIO()
    Image.new("L", size, value).save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_script_uses_frontal_rows_from_fake_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "tiny.zip"
    rows = [
        {"Path": "CheXpert-v1.0-small/train/patient1/study1/view1.png", "Frontal/Lateral": "Frontal"},
        {"Path": "CheXpert-v1.0-small/train/patient2/study1/view1.png", "Frontal/Lateral": "Lateral"},
        {"Path": "CheXpert-v1.0-small/train/patient3/study1/view1.png", "Frontal/Lateral": "Frontal"},
    ]
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("CheXpert-v1.0-small/train.csv", pd.DataFrame(rows).to_csv(index=False))
        for row in rows:
            archive.writestr(row["Path"], _png_bytes((8, 6)))
    output_dir = tmp_path / "subset"
    result = subprocess.run(
        [sys.executable, "scripts/extract_chexpert_image_subset.py", "--zip-path", str(zip_path),
         "--metadata-csv", "CheXpert-v1.0-small/train.csv", "--output-dir", str(output_dir), "--limit", "2"],
        cwd=REPO_ROOT, env=_environment(), check=True, capture_output=True, text=True,
    )
    manifest = output_dir / "subset_metadata.csv"
    assert str(manifest) in result.stdout
    frame = pd.read_csv(manifest)
    assert frame["Frontal/Lateral"].tolist() == ["Frontal", "Frontal"]
    assert all((output_dir / path).is_file() for path in frame["image_path"])


def test_audit_script_summarizes_tiny_fake_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (10, 20)).save(image_dir / "one.png")
    Image.new("L", (20, 30)).save(image_dir / "two.png")
    (image_dir / "bad.png").write_text("not an image", encoding="utf-8")
    output = tmp_path / "qc.json"
    subprocess.run(
        [sys.executable, "scripts/audit_chexpert_images.py", "--image-dir", str(image_dir), "--output-json", str(output)],
        cwd=REPO_ROOT, env=_environment(), check=True, capture_output=True, text=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["n_total"] == 3
    assert summary["n_readable"] == 2
    assert summary["n_unreadable"] == 1
    assert summary["mode_counts"] == {"L": 2}
    assert summary["width_mean"] == 15.0


def _toy_predictions() -> pd.DataFrame:
    rows = []
    targets = [0, 1, -1, None]
    scores = [0.1, 0.9, 0.5, 0.7]
    mapping = {
        "Atelectasis": "pred_Atelectasis",
        "Cardiomegaly": "pred_Cardiomegaly",
        "Pleural Effusion": "pred_Effusion",
        "Pneumonia": "pred_Pneumonia",
        "Pneumothorax": "pred_Pneumothorax",
    }
    for index in range(4):
        row = {}
        for label, prediction in mapping.items():
            row[label] = targets[index]
            row[prediction] = scores[index]
        rows.append(row)
    return pd.DataFrame(rows)


def test_evaluation_script_writes_toy_metrics(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _toy_predictions().to_csv(predictions, index=False)
    output_dir = tmp_path / "results"
    subprocess.run(
        [sys.executable, "scripts/evaluate_chexpert_image_predictions.py", "--predictions-csv", str(predictions), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, env=_environment(), check=True, capture_output=True, text=True,
    )
    metrics = pd.read_csv(output_dir / "evaluation_label_metrics.csv")
    assert len(metrics) == 5
    assert set(metrics["n_available_soft"]) == {3}
    assert set(metrics["n_available_binary"]) == {2}
    assert set(metrics["auroc_binary"]) == {1.0}
    report = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["evaluation_policy"]["threshold_tuning"] is False


def test_plotting_script_creates_four_metric_figures(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [{"label": "Toy", "auroc_binary": 0.8, "auprc_binary": 0.7, "brier_soft": 0.2,
          "ece_soft": 0.1, "sensitivity_at_0_5": 0.9, "specificity_at_0_5": 0.6}]
    ).to_csv(metrics_path, index=False)
    output_dir = tmp_path / "figures"
    subprocess.run(
        [sys.executable, "scripts/plot_chexpert_image_inference_results.py", "--metrics-csv", str(metrics_path), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, env=_environment(), check=True, capture_output=True, text=True,
    )
    assert {path.name for path in output_dir.glob("*.png")} == {
        "auroc_by_label.png", "auprc_by_label.png", "calibration_metrics.png",
        "sensitivity_specificity_at_0_5.png",
    }


def test_inference_script_reports_missing_torchxrayvision(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.mkdir()
    (blocker / "torchxrayvision.py").write_text("raise ImportError('blocked for test')\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"image_path": ["image.png"]}).to_csv(manifest, index=False)
    result = subprocess.run(
        [sys.executable, "scripts/run_torchxrayvision_inference.py", "--manifest-csv", str(manifest),
         "--image-root", str(tmp_path), "--output-csv", str(tmp_path / "predictions.csv")],
        cwd=REPO_ROOT, env=_environment(blocker), check=False, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "torchxrayvision is required for inference" in result.stderr
