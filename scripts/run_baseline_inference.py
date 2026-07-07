#!/usr/bin/env python3
"""Run bounded local baseline TorchXRayVision inference on a small manifest."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Sequence

from medshiftlab.data import load_inference_manifest_csv, load_local_data_config
from medshiftlab.models import TorchXRayVisionAdapter, TorchXRayVisionAdapterConfig
from medshiftlab.reporting import (
    write_prediction_batch_json,
    write_prediction_records_csv,
)


DEFAULT_LIMIT = 16
MAX_SAFE_LIMIT_WITHOUT_OVERRIDE = 64
PROJECT_LABEL_TO_XRV_PATHOLOGY = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Pleural Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the ignored local dataset-path YAML config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Canonical dataset name used by the registry and manifest.",
    )
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="CSV manifest with sample_id/image_id and relative image_path values.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional JSON output path for standardized predictions.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional CSV output path for standardized predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of manifest rows to process. Default: {DEFAULT_LIMIT}.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=(
            "Allow limits above the default safe subset threshold. "
            "Use only for explicit local/manual runs."
        ),
    )
    parser.add_argument(
        "--allow-model-init",
        action="store_true",
        help=(
            "Allow local model initialization. This is disabled by default so the "
            "script does not create a weight-resolution path unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--weights",
        default="densenet121-res224-all",
        help="TorchXRayVision weight identifier for explicit local model init.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def run_baseline_inference(args: argparse.Namespace) -> dict[str, object]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if (
        args.limit > MAX_SAFE_LIMIT_WITHOUT_OVERRIDE
        and not args.allow_large_run
    ):
        raise ValueError(
            f"--limit {args.limit} exceeds the safe manual subset threshold of "
            f"{MAX_SAFE_LIMIT_WITHOUT_OVERRIDE}. Re-run with --allow-large-run only "
            "for an explicit local/manual execution."
        )
    if not args.allow_model_init:
        raise RuntimeError(
            "Model initialization is disabled by default. Re-run with "
            "--allow-model-init only after optional dependencies are installed and "
            "authorized local model weights are available."
        )

    local_data_config = load_local_data_config(args.config)
    manifest_records = load_inference_manifest_csv(
        args.manifest_csv,
        dataset_name=args.dataset,
        limit=args.limit,
    )
    model = _initialize_torchxrayvision_model(
        weights=args.weights,
        device=args.device,
    )
    output_indices = _build_output_indices(model)
    adapter = TorchXRayVisionAdapter(
        config=TorchXRayVisionAdapterConfig(
            model_name="torchxrayvision-densenet121",
            model_version=f"torchxrayvision:{args.weights}",
            labels=tuple(PROJECT_LABEL_TO_XRV_PATHOLOGY),
            output_indices=output_indices,
            device=args.device,
            preprocessing_config={
                "manual_only": True,
                "safe_subset_limit": MAX_SAFE_LIMIT_WITHOUT_OVERRIDE,
            },
            batch_size=args.batch_size,
            output_activation="sigmoid",
        ),
        model=model,
        local_data_config=local_data_config,
    )
    prediction_batch = adapter.predict_records(
        [
            {
                "sample_id": record.sample_id,
                "patient_id": record.patient_id,
                "study_id": record.study_id,
                "dataset_name": record.dataset_name,
                "image_path": record.image_path,
            }
            for record in manifest_records
        ]
    )

    written_outputs: dict[str, str] = {}
    if args.output_json:
        written_outputs["json"] = str(
            write_prediction_batch_json(prediction_batch, args.output_json)
        )
    if args.output_csv:
        written_outputs["csv"] = str(
            write_prediction_records_csv(prediction_batch, args.output_csv)
        )

    return {
        "dataset_name": args.dataset,
        "model_name": prediction_batch.model_name,
        "model_version": prediction_batch.model_version,
        "adapter_name": prediction_batch.adapter_name,
        "schema_version": prediction_batch.schema_version,
        "n_records": len(prediction_batch.records),
        "limit": args.limit,
        "outputs_written": written_outputs,
        "manual_only": True,
        "full_benchmark_completed": False,
        "external_validation_completed": False,
        "clinical_validation_completed": False,
    }


def _initialize_torchxrayvision_model(*, weights: str, device: str) -> object:
    torch, xrv = _load_runtime_dependencies()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    model = xrv.models.DenseNet(weights=weights).to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _load_runtime_dependencies() -> tuple[object, object]:
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for baseline inference. Install the "
            "'torchxrayvision' optional dependencies."
        ) from error
    try:
        xrv = import_module("torchxrayvision")
    except ImportError as error:
        raise RuntimeError(
            "torchxrayvision is required for baseline inference. Install the "
            "'torchxrayvision' optional dependencies."
        ) from error
    return torch, xrv


def _build_output_indices(model: object) -> dict[str, int]:
    pathologies = getattr(model, "pathologies", None)
    if not isinstance(pathologies, Sequence):
        raise RuntimeError("Initialized model does not expose a pathologies sequence")

    available = {str(pathology): index for index, pathology in enumerate(pathologies)}
    output_indices: dict[str, int] = {}
    missing: list[str] = []
    for label_name, pathology_name in PROJECT_LABEL_TO_XRV_PATHOLOGY.items():
        index = available.get(pathology_name)
        if index is None:
            missing.append(f"{label_name}<-{pathology_name}")
            continue
        output_indices[label_name] = index

    if missing:
        raise RuntimeError(
            "Initialized model is missing required pathologies for the project label "
            "mapping: " + ", ".join(missing)
        )
    return output_indices


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_baseline_inference(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Baseline inference error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
