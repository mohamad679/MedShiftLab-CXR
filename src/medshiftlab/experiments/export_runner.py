"""File-exporting wrapper for in-memory evaluation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from medshiftlab.experiments.runner import (
    InMemoryExperimentConfig,
    InMemoryExperimentResult,
    run_in_memory_evaluation_experiment,
)
from medshiftlab.models.adapter import CXRModelAdapter
from medshiftlab.reporting.evaluation_export import write_evaluation_report_bundle


class _ExperimentRecord(Protocol):
    image_id: str
    labels: Mapping[str, float | None]


class ExportedExperimentResult(BaseModel):
    """In-memory experiment result and paths to its report exports."""

    model_config = ConfigDict(extra="forbid")

    experiment_result: InMemoryExperimentResult
    output_dir: Path
    output_paths: dict[str, Path] = Field(min_length=1)


def run_and_export_evaluation_experiment(
    records: Sequence[_ExperimentRecord | Mapping[str, object]],
    adapter: CXRModelAdapter,
    config: InMemoryExperimentConfig,
    output_dir: str | Path,
    *,
    json_filename: str = "evaluation_report.json",
    csv_filename: str = "evaluation_label_metrics.csv",
    target_prefix: str = "true_",
    score_prefix: str = "score_",
) -> ExportedExperimentResult:
    """Run an in-memory evaluation and export only its report bundle."""

    experiment_result = run_in_memory_evaluation_experiment(
        records=records,
        adapter=adapter,
        config=config,
        target_prefix=target_prefix,
        score_prefix=score_prefix,
    )
    directory = Path(output_dir)
    bundle_paths = write_evaluation_report_bundle(
        experiment_result.report,
        directory,
        json_filename=json_filename,
        csv_filename=csv_filename,
    )

    return ExportedExperimentResult(
        experiment_result=experiment_result,
        output_dir=directory,
        output_paths={
            "json": bundle_paths["json"],
            "label_metrics_csv": bundle_paths["csv"],
        },
    )
