#!/usr/bin/env python3
"""Extract a deterministic frontal CheXpert image subset from an archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--metadata-csv", required=True, help="CSV path or member name in the zip.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--manifest-name", default="subset_metadata.csv")
    return parser


def _safe_destination(root: Path, member: str) -> Path:
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member path: {member}")
    return root.joinpath(*relative.parts)


def _read_metadata(archive: zipfile.ZipFile, csv_reference: str) -> pd.DataFrame:
    csv_path = Path(csv_reference)
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    try:
        with archive.open(csv_reference) as handle:
            return pd.read_csv(handle)
    except KeyError as error:
        raise FileNotFoundError(
            f"Metadata CSV is neither a local file nor a zip member: {csv_reference}"
        ) from error


def extract_subset(
    zip_path: Path,
    metadata_csv: str,
    output_dir: Path,
    limit: int,
    manifest_name: str,
) -> Path:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        metadata = _read_metadata(archive, metadata_csv)
        required = {"Path", "Frontal/Lateral"}
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError("Missing metadata columns: " + ", ".join(missing))
        subset = metadata.loc[metadata["Frontal/Lateral"] == "Frontal"].head(limit).copy()
        if len(subset) < limit:
            raise ValueError(f"Requested {limit} frontal images but found {len(subset)}")
        names = set(archive.namelist())
        for member in tqdm(subset["Path"], desc="Extracting images", unit="image"):
            if member not in names:
                raise FileNotFoundError(f"Image member missing from zip: {member}")
            destination = _safe_destination(output_dir, member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
    subset.insert(0, "image_path", subset["Path"].astype(str))
    manifest = output_dir / manifest_name
    subset.to_csv(manifest, index=False)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = extract_subset(
        Path(args.zip_path), args.metadata_csv, Path(args.output_dir), args.limit, args.manifest_name
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
