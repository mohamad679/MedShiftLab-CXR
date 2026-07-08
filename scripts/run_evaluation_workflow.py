#!/usr/bin/env python3
"""Run the MedShiftLab local evaluation workflow.

The workflow orchestrates the reusable local/manual evaluation scripts:
1. threshold sweep
2. calibrated threshold evaluation
3. bootstrap uncertainty
4. subgroup audit

This is exploratory tooling only. It is not clinical validation, external
validation, fairness validation, or full benchmarking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _outputs(output_dir: Path) -> dict[str, Path]:
    return {
        "threshold_sweep_json": output_dir / "threshold_sweep.json",
        "threshold_sweep_csv": output_dir / "threshold_sweep.csv",
        "threshold_sweep_best_csv": output_dir / "threshold_sweep_best.csv",
        "calibrated_eval_json": output_dir / "calibrated_threshold_eval.json",
        "calibration_best_csv": output_dir / "calibration_best.csv",
        "evaluation_csv": output_dir / "evaluation.csv",
        "split_csv": output_dir / "split.csv",
        "bootstrap_json": output_dir / "bootstrap_uncertainty.json",
        "bootstrap_csv": output_dir / "bootstrap_uncertainty.csv",
        "subgroup_json": output_dir / "subgroup_audit.json",
        "subgroup_label_metrics_csv": output_dir / "subgroup_label_metrics.csv",
        "subgroup_aggregate_csv": output_dir / "subgroup_aggregate.csv",
    }


def _cmds(args: argparse.Namespace, outputs: dict[str, Path]) -> list[list[str]]:
    repo = _repo_root()
    common_notes = args.notes_prefix

    return [
        [
            sys.executable,
            str(repo / "scripts/run_threshold_sweep.py"),
            "--predictions",
            str(args.predictions),
            "--labels-csv",
            str(args.labels_csv),
            "--output-json",
            str(outputs["threshold_sweep_json"]),
            "--output-csv",
            str(outputs["threshold_sweep_csv"]),
            "--output-best-csv",
            str(outputs["threshold_sweep_best_csv"]),
            "--threshold-start",
            str(args.threshold_start),
            "--threshold-stop",
            str(args.threshold_stop),
            "--threshold-step",
            str(args.threshold_step),
            "--notes",
            f"{common_notes}: threshold sweep",
        ],
        [
            sys.executable,
            str(repo / "scripts/run_calibrated_threshold_evaluation.py"),
            "--predictions",
            str(args.predictions),
            "--labels-csv",
            str(args.labels_csv),
            "--output-json",
            str(outputs["calibrated_eval_json"]),
            "--output-calibration-best-csv",
            str(outputs["calibration_best_csv"]),
            "--output-evaluation-csv",
            str(outputs["evaluation_csv"]),
            "--output-split-csv",
            str(outputs["split_csv"]),
            "--calibration-fraction",
            str(args.calibration_fraction),
            "--threshold-start",
            str(args.threshold_start),
            "--threshold-stop",
            str(args.threshold_stop),
            "--threshold-step",
            str(args.threshold_step),
            "--notes",
            f"{common_notes}: calibrated threshold evaluation",
        ],
        [
            sys.executable,
            str(repo / "scripts/run_bootstrap_uncertainty.py"),
            "--predictions",
            str(args.predictions),
            "--labels-csv",
            str(args.labels_csv),
            "--split-csv",
            str(outputs["split_csv"]),
            "--calibrated-threshold-eval",
            str(outputs["calibrated_eval_json"]),
            "--output-json",
            str(outputs["bootstrap_json"]),
            "--output-csv",
            str(outputs["bootstrap_csv"]),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--notes",
            f"{common_notes}: bootstrap uncertainty",
        ],
        [
            sys.executable,
            str(repo / "scripts/run_subgroup_audit.py"),
            "--predictions",
            str(args.predictions),
            "--labels-csv",
            str(args.labels_csv),
            "--split-csv",
            str(outputs["split_csv"]),
            "--calibrated-threshold-eval",
            str(outputs["calibrated_eval_json"]),
            "--metadata-csv",
            str(args.metadata_csv),
            "--subgroup-columns",
            *args.subgroup_columns,
            "--output-json",
            str(outputs["subgroup_json"]),
            "--output-label-metrics-csv",
            str(outputs["subgroup_label_metrics_csv"]),
            "--output-aggregate-csv",
            str(outputs["subgroup_aggregate_csv"]),
            "--notes",
            f"{common_notes}: subgroup audit",
        ],
    ]


def _command_string(command: list[str]) -> str:
    return " ".join(command)


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = _outputs(args.output_dir)
    commands = _cmds(args, output_paths)

    payload: dict[str, Any] = {
        "schema_version": "medshiftlab.evaluation_workflow.v1",
        "execute": bool(args.execute),
        "repo_root": str(_repo_root()),
        "inputs": {
            "predictions": str(args.predictions),
            "labels_csv": str(args.labels_csv),
            "metadata_csv": str(args.metadata_csv),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "settings": {
            "calibration_fraction": args.calibration_fraction,
            "threshold_start": args.threshold_start,
            "threshold_stop": args.threshold_stop,
            "threshold_step": args.threshold_step,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "subgroup_columns": args.subgroup_columns,
        },
        "commands": [_command_string(command) for command in commands],
        "notes": (
            "Exploratory local/manual evaluation workflow. Not clinical validation, "
            "not external validation, not fairness validation, and not a full benchmark."
        ),
    }

    if not args.execute:
        return payload

    _require_existing(args.predictions, "predictions")
    _require_existing(args.labels_csv, "labels_csv")
    _require_existing(args.metadata_csv, "metadata_csv")
    if not args.subgroup_columns:
        raise ValueError("At least one subgroup column is required.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("\n$", _command_string(command))
        subprocess.run(command, cwd=_repo_root(), check=True)

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--subgroup-columns", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-fraction", default=0.5, type=float)
    parser.add_argument("--threshold-start", default=0.0, type=float)
    parser.add_argument("--threshold-stop", default=1.0, type=float)
    parser.add_argument("--threshold-step", default=0.01, type=float)
    parser.add_argument("--n-bootstrap", default=1000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260708, type=int)
    parser.add_argument("--notes-prefix", default="MedShiftLab evaluation workflow")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_workflow(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
