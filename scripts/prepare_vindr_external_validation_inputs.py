#!/usr/bin/env python3
"""Prepare VinDr-CXR-style external-validation inputs for MedShiftLab-CXR.

This scaffold converts a VinDr-style annotation table into local/private
MedShiftLab-compatible labels, optional metadata, optional manifest, and a
summary JSON. It does not download data and does not perform inference,
clinical validation, external validation, fairness validation, or full
benchmarking.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_mapping(mapping_json_path: Path) -> dict[str, Any]:
    mapping = json.loads(mapping_json_path.read_text(encoding="utf-8"))

    required = {"target_labels", "source_to_target"}
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"Mapping JSON is missing required keys: {missing}")

    target_labels = list(mapping["target_labels"])
    source_to_target = dict(mapping["source_to_target"])

    bad_targets = sorted(set(source_to_target.values()) - set(target_labels))
    if bad_targets:
        raise ValueError(f"Mapping JSON contains targets not in target_labels: {bad_targets}")

    normalized_source_to_target = {
        _normalize_label(source): target for source, target in source_to_target.items()
    }

    mapping["_normalized_source_to_target"] = normalized_source_to_target
    return mapping


def _build_labels(
    *,
    annotations: pd.DataFrame,
    mapping: dict[str, Any],
    image_id_column: str,
    class_column: str,
    sample_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if image_id_column not in annotations.columns:
        raise ValueError(f"annotations CSV is missing image id column: {image_id_column}")
    if class_column not in annotations.columns:
        raise ValueError(f"annotations CSV is missing class column: {class_column}")

    target_labels = list(mapping["target_labels"])
    normalized_source_to_target = dict(mapping["_normalized_source_to_target"])

    work = annotations[[image_id_column, class_column]].copy()
    work[image_id_column] = work[image_id_column].astype(str).str.strip()
    work[class_column] = work[class_column].astype(str).str.strip()
    work = work[work[image_id_column] != ""].copy()

    image_ids = sorted(work[image_id_column].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []

    mapped_counts = {label: 0 for label in target_labels}
    unmapped_counts: dict[str, int] = {}

    grouped = work.groupby(image_id_column, sort=True)
    for image_id in image_ids:
        row: dict[str, Any] = {
            "sample_id": f"{sample_prefix}_{image_id}",
            "source_image_id": image_id,
        }
        for label in target_labels:
            row[label] = 0.0

        for source_label in grouped.get_group(image_id)[class_column].dropna().tolist():
            normalized = _normalize_label(source_label)
            target = normalized_source_to_target.get(normalized)
            if target is None:
                unmapped_counts[str(source_label)] = unmapped_counts.get(str(source_label), 0) + 1
                continue
            row[target] = 1.0

        for label in target_labels:
            mapped_counts[label] += int(row[label] == 1.0)

        rows.append(row)

    labels = pd.DataFrame(rows)
    summary = {
        "n_annotation_rows": int(len(annotations)),
        "n_unique_images": int(len(labels)),
        "target_labels": target_labels,
        "positive_counts": mapped_counts,
        "unmapped_source_label_counts": dict(sorted(unmapped_counts.items())),
    }
    return labels, summary


def _prepare_metadata(
    *,
    metadata_csv_path: Path | None,
    image_id_column: str,
    sample_prefix: str,
    metadata_columns: list[str],
) -> pd.DataFrame | None:
    if metadata_csv_path is None:
        return None

    metadata = pd.read_csv(metadata_csv_path)
    if image_id_column not in metadata.columns:
        raise ValueError(f"metadata CSV is missing image id column: {image_id_column}")

    columns = [image_id_column]
    for column in metadata_columns:
        if column not in metadata.columns:
            raise ValueError(f"metadata CSV is missing requested column: {column}")
        columns.append(column)

    out = metadata[columns].copy()
    out[image_id_column] = out[image_id_column].astype(str).str.strip()
    out.insert(0, "sample_id", sample_prefix + "_" + out[image_id_column].astype(str))
    out = out.rename(columns={image_id_column: "source_image_id"})
    return out


def _find_image_path(image_root: Path, source_image_id: str, extensions: list[str]) -> tuple[str, bool]:
    for ext in extensions:
        normalized_ext = ext if ext.startswith(".") else f".{ext}"
        candidate = image_root / f"{source_image_id}{normalized_ext}"
        if candidate.is_file():
            return str(candidate.relative_to(image_root)), True
    return "", False


def _prepare_manifest(
    *,
    labels: pd.DataFrame,
    dataset_name: str,
    image_root: Path | None,
    image_extensions: list[str],
) -> pd.DataFrame | None:
    if image_root is None:
        return None

    rows: list[dict[str, Any]] = []
    for row in labels.to_dict(orient="records"):
        image_path, image_found = _find_image_path(
            image_root=image_root,
            source_image_id=str(row["source_image_id"]),
            extensions=image_extensions,
        )
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": dataset_name,
                "image_path": image_path,
                "source_image_id": row["source_image_id"],
                "image_found": bool(image_found),
            }
        )
    return pd.DataFrame(rows)


def prepare_vindr_inputs(
    *,
    annotations_csv_path: Path,
    mapping_json_path: Path,
    output_dir: Path,
    image_id_column: str = "image_id",
    class_column: str = "class_name",
    metadata_csv_path: Path | None = None,
    metadata_columns: list[str] | None = None,
    image_root: Path | None = None,
    image_extensions: list[str] | None = None,
    dataset_name: str = "vindr_cxr",
    sample_prefix: str = "vindr",
) -> dict[str, Any]:
    metadata_columns = metadata_columns or []
    image_extensions = image_extensions or [".jpg", ".jpeg", ".png", ".dicom", ".dcm"]

    annotations = pd.read_csv(annotations_csv_path)
    mapping = _load_mapping(mapping_json_path)

    labels, label_summary = _build_labels(
        annotations=annotations,
        mapping=mapping,
        image_id_column=image_id_column,
        class_column=class_column,
        sample_prefix=sample_prefix,
    )

    metadata = _prepare_metadata(
        metadata_csv_path=metadata_csv_path,
        image_id_column=image_id_column,
        sample_prefix=sample_prefix,
        metadata_columns=metadata_columns,
    )

    manifest = _prepare_manifest(
        labels=labels,
        dataset_name=dataset_name,
        image_root=image_root,
        image_extensions=image_extensions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "vindr_labels.csv"
    metadata_path = output_dir / "vindr_metadata.csv"
    manifest_path = output_dir / "vindr_manifest.csv"
    summary_path = output_dir / "vindr_prepare_summary.json"

    labels.to_csv(labels_path, index=False)

    outputs: dict[str, str | None] = {
        "labels_csv": str(labels_path),
        "metadata_csv": None,
        "manifest_csv": None,
        "summary_json": str(summary_path),
    }

    metadata_summary: dict[str, Any] | None = None
    if metadata is not None:
        metadata.to_csv(metadata_path, index=False)
        outputs["metadata_csv"] = str(metadata_path)
        metadata_summary = {
            "n_metadata_rows": int(len(metadata)),
            "metadata_columns": list(metadata.columns),
        }

    manifest_summary: dict[str, Any] | None = None
    if manifest is not None:
        manifest.to_csv(manifest_path, index=False)
        outputs["manifest_csv"] = str(manifest_path)
        manifest_summary = {
            "n_manifest_rows": int(len(manifest)),
            "n_images_found": int(manifest["image_found"].sum()),
            "n_images_missing": int((~manifest["image_found"]).sum()),
            "image_root": str(image_root),
        }

    payload = {
        "schema_version": "medshiftlab.vindr_external_validation_inputs.v1",
        "dataset_name": dataset_name,
        "claim_level": {
            "external_validation_completed": False,
            "clinical_validation_completed": False,
            "full_benchmark_completed": False,
            "fairness_validation_completed": False,
        },
        "source_files": {
            "annotations_csv": str(annotations_csv_path),
            "metadata_csv": str(metadata_csv_path) if metadata_csv_path else None,
            "mapping_json": str(mapping_json_path),
        },
        "label_summary": label_summary,
        "metadata_summary": metadata_summary,
        "manifest_summary": manifest_summary,
        "outputs": outputs,
        "notes": (
            "VinDr external-validation scaffold input preparation only. "
            "No inference, external validation, clinical validation, fairness validation, or full benchmark was performed."
        ),
    }

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True, type=Path)
    parser.add_argument(
        "--mapping-json",
        default=Path("configs/evaluation/vindr_cxr_label_mapping.json"),
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--class-column", default="class_name")
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--metadata-columns", nargs="*", default=[])
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--image-extensions",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".dicom", ".dcm"],
    )
    parser.add_argument("--dataset-name", default="vindr_cxr")
    parser.add_argument("--sample-prefix", default="vindr")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = prepare_vindr_inputs(
        annotations_csv_path=args.annotations_csv,
        mapping_json_path=args.mapping_json,
        output_dir=args.output_dir,
        image_id_column=args.image_id_column,
        class_column=args.class_column,
        metadata_csv_path=args.metadata_csv,
        metadata_columns=list(args.metadata_columns),
        image_root=args.image_root,
        image_extensions=list(args.image_extensions),
        dataset_name=args.dataset_name,
        sample_prefix=args.sample_prefix,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
