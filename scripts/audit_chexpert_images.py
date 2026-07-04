#!/usr/bin/env python3
"""Audit readability, modes, and dimensions of a local image subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Sequence

from PIL import Image
from tqdm.auto import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def audit_images(image_dir: Path) -> dict[str, object]:
    paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    modes: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    for path in tqdm(paths, desc="Auditing images", unit="image"):
        try:
            with Image.open(path) as image:
                image.load()
                modes[image.mode] += 1
                widths.append(image.width)
                heights.append(image.height)
        except (OSError, ValueError):
            continue
    summary: dict[str, object] = {
        "n_total": len(paths),
        "n_readable": len(widths),
        "n_unreadable": len(paths) - len(widths),
        "mode_counts": dict(sorted(modes.items())),
    }
    for name, values in (("width", widths), ("height", heights)):
        summary[f"{name}_min"] = float(min(values)) if values else None
        summary[f"{name}_max"] = float(max(values)) if values else None
        summary[f"{name}_mean"] = float(mean(values)) if values else None
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_images(Path(args.image_dir)), indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
