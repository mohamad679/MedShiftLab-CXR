#!/usr/bin/env python3
"""Run a bounded, read-only image-loading smoke test from local registry config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from medshiftlab.data import (
    SUPPORTED_DATASET_NAMES,
    ImagePreprocessingConfig,
    discover_dataset_image_paths,
    load_local_data_config,
    summarize_dataset_images,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ignored local-path YAML file")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASET_NAMES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument(
        "--normalization",
        choices=("none", "zero_one", "minus_one_one", "standardize"),
        default="zero_one",
    )
    parser.add_argument("--standardize-mean", type=float, default=0.0)
    parser.add_argument("--standardize-std", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        preprocessing = ImagePreprocessingConfig(
            output_mode=args.output_mode,
            target_size=(args.height, args.width),
            normalization=args.normalization,
            standardize_mean=args.standardize_mean,
            standardize_std=args.standardize_std,
        )
        config = load_local_data_config(args.config)
        image_paths = discover_dataset_image_paths(
            config,
            args.dataset,
            limit=args.limit,
        )
        summary = summarize_dataset_images(
            config,
            args.dataset,
            image_paths,
            preprocessing,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps(summary.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
