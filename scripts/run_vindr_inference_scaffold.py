#!/usr/bin/env python3
"""VinDr/VinBigData inference-runner scaffold.

This script validates prepared VinDr manifest/label inputs and writes a
local/private inference-readiness summary. It does not load a model, does not
run image inference, does not create predictions, and does not compute metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def _read_csv(path: Path, *, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"{name} is empty: {path}")
    return data


def _validate_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    required = {"sample_id", "dataset_name", "image_path", "source_image_id", "image_found"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    n_rows = int(len(manifest))
    n_images_found = int(manifest["image_found"].astype(bool).sum())
    n_images_missing = int((~manifest["image_found"].astype(bool)).sum())

    if n_rows == 0:
        raise ValueError("manifest has zero rows")
    if n_images_found == 0:
        raise ValueError("manifest has zero found images")
    if n_images_missing > 0:
        raise ValueError(f"manifest contains missing images: {n_images_missing}")

    absolute_paths = manifest["image_path"].astype(str).str.startswith("/").sum()
    if absolute_paths:
        raise ValueError("manifest image_path values must be relative, not absolute")

    return {
        "n_manifest_rows": n_rows,
        "n_images_found": n_images_found,
        "n_images_missing": n_images_missing,
    }


def _validate_labels(labels: pd.DataFrame, target_labels: list[str]) -> dict[str, Any]:
    required = {"sample_id", "source_image_id", *target_labels}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels are missing required columns: {missing}")

    return {
        "n_label_rows": int(len(labels)),
        "target_labels": target_labels,
        "positive_counts": {
            label: int(labels[label].fillna(0).astype(float).sum()) for label in target_labels
        },
    }


def build_inference_scaffold_summary(
    *,
    manifest_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    target_labels: list[str] | None = None,
    run_name: str = "vindr_inference_scaffold",
) -> dict[str, Any]:
    target_labels = target_labels or DEFAULT_TARGET_LABELS

    manifest = _read_csv(manifest_csv, name="manifest CSV")
    labels = _read_csv(labels_csv, name="labels CSV")

    manifest_summary = _validate_manifest(manifest)
    labels_summary = _validate_labels(labels, target_labels)

    manifest_samples = set(manifest["sample_id"].astype(str))
    label_samples = set(labels["sample_id"].astype(str))
    missing_label_rows = sorted(manifest_samples - label_samples)
    extra_label_rows = sorted(label_samples - manifest_samples)

    if missing_label_rows:
        raise ValueError(f"labels are missing manifest sample rows: {len(missing_label_rows)}")
    if extra_label_rows:
        raise ValueError(f"labels contain rows not present in manifest: {len(extra_label_rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "vindr_inference_scaffold_summary.json"

    payload = {
        "schema_version": "medshiftlab.vindr_inference_scaffold.v1",
        "run_name": run_name,
        "claim_level": {
            "model_loaded": False,
            "inference_completed": False,
            "external_validation_completed": False,
            "clinical_validation_completed": False,
            "metrics_completed": False,
        },
        "input_summary": {
            "manifest": manifest_summary,
            "labels": labels_summary,
            "n_joinable_samples": int(len(manifest_samples & label_samples)),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "predictions_csv": None,
            "metrics_json": None,
        },
        "notes": (
            "Inference-runner scaffold only. No model was loaded, no image inference was run, "
            "no predictions were generated, and no metrics were computed."
        ),
    }

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-labels", nargs="*", default=DEFAULT_TARGET_LABELS)
    parser.add_argument("--run-name", default="vindr_inference_scaffold")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_inference_scaffold_summary(
        manifest_csv=args.manifest_csv,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        target_labels=list(args.target_labels),
        run_name=args.run_name,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
