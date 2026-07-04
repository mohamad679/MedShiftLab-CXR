#!/usr/bin/env python3
"""Run pretrained TorchXRayVision DenseNet inference on a CheXpert subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _load_dependencies():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for inference. Install the 'torchxrayvision' optional dependencies."
        ) from error
    try:
        import torchxrayvision as xrv
    except ImportError as error:
        raise RuntimeError(
            "torchxrayvision is required for inference. Install the 'torchxrayvision' optional dependencies."
        ) from error
    return torch, xrv


def _resolve_image(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Manifest image_path must be relative and safe: {relative_path}")
    return root / candidate


def run_inference(args: argparse.Namespace) -> Path:
    torch, xrv = _load_dependencies()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")

    manifest = pd.read_csv(args.manifest_csv)
    if "image_path" not in manifest:
        raise ValueError("Manifest must contain an image_path column")
    image_root = Path(args.image_root)
    model = xrv.models.DenseNet(weights=args.weights).to(args.device)
    model.eval()
    center_crop = xrv.datasets.XRayCenterCrop()
    resize = xrv.datasets.XRayResizer(224)

    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(manifest), args.batch_size), desc="Inference", unit="batch"):
        tensors = []
        for relative_path in manifest["image_path"].iloc[start : start + args.batch_size]:
            image_path = _resolve_image(image_root, str(relative_path))
            with Image.open(image_path) as image:
                array = np.asarray(image.convert("L"), dtype=np.float32)
            array = xrv.datasets.normalize(array, 255)
            array = center_crop(array[None, ...])
            array = resize(array)
            tensors.append(torch.from_numpy(array))
        batch = torch.stack(tensors).to(args.device)
        with torch.inference_mode():
            batches.append(model(batch).detach().cpu().numpy())

    predictions = np.concatenate(batches, axis=0) if batches else np.empty((0, len(model.pathologies)))
    if predictions.shape[1] != len(model.pathologies):
        raise RuntimeError("Model output width does not match model.pathologies")
    output = manifest.copy()
    for index, pathology in enumerate(model.pathologies):
        output[f"pred_{pathology}"] = predictions[:, index]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"prediction_shape={predictions.shape}")
    print(output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_inference(args)
    except RuntimeError as error:
        print(f"Inference dependency/runtime error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
