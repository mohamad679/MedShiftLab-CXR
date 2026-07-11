"""Tests for robustness report aggregate export helpers."""

from __future__ import annotations

from pathlib import Path

from medshiftlab.evaluation import summarize_label_calibration
from medshiftlab.evaluation.robustness import (
    FailureCaseSummary,
    RobustnessAnalysisReport,
    SubgroupAnalysisReport,
)
from medshiftlab.reporting import (
    write_calibration_curves_png,
    write_robustness_report_bundle,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _robustness_report(
    *,
    include_plottable_label: bool = True,
) -> RobustnessAnalysisReport:
    calibration = {
        "NoData": summarize_label_calibration(
            "NoData",
            [None, None],
            [None, None],
            n_bins=4,
        )
    }
    if include_plottable_label:
        calibration["Finding"] = summarize_label_calibration(
            "Finding",
            [0, 0, 1, 1],
            [0.1, 0.3, 0.7, 0.9],
            n_bins=4,
        )

    return RobustnessAnalysisReport(
        dataset_name="synthetic",
        model_name="mock",
        evaluated_records=4,
        calibration=calibration,
        bootstrap_intervals=[],
        subgroup_analysis=SubgroupAnalysisReport(
            subgroup_columns=("dataset_name",),
            minimum_subgroup_size=1,
            results=[],
            skipped=[],
            coverage=[],
        ),
        failure_cases=FailureCaseSummary(
            metric_degradations=[],
            poor_calibration=[],
            coverage_issues=[],
        ),
        support_status={
            "calibration_bin_data": "implemented",
            "calibration_curve_csv": "implemented",
            "calibration_plot_export": "implemented_optional",
        },
    )


def test_calibration_curves_png_is_written_and_creates_parent_dirs(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "plots" / "calibration_curves.png"

    written = write_calibration_curves_png(_robustness_report(), output_path)

    assert written == output_path
    assert output_path.exists()
    assert output_path.read_bytes().startswith(PNG_SIGNATURE)


def test_calibration_curves_png_skips_empty_bins_without_crashing(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "empty_only" / "calibration_curves.png"

    written = write_calibration_curves_png(
        _robustness_report(include_plottable_label=False),
        output_path,
    )

    assert written == output_path
    assert output_path.read_bytes().startswith(PNG_SIGNATURE)


def test_bundle_includes_calibration_plot_only_when_requested(
    tmp_path: Path,
) -> None:
    report = _robustness_report()

    default_outputs = write_robustness_report_bundle(report, tmp_path / "default")
    assert "calibration_plot" not in default_outputs
    assert not (tmp_path / "default" / "calibration_curves.png").exists()

    plot_outputs = write_robustness_report_bundle(
        report,
        tmp_path / "with_plot",
        export_calibration_plot=True,
    )

    assert plot_outputs["calibration_plot"].name == "calibration_curves.png"
    assert plot_outputs["calibration_plot"].read_bytes().startswith(PNG_SIGNATURE)


def test_robustness_exports_do_not_write_private_paths_or_image_path(
    tmp_path: Path,
) -> None:
    outputs = write_robustness_report_bundle(
        _robustness_report(),
        tmp_path / "bundle",
        export_calibration_csv=True,
        export_calibration_plot=True,
    )

    for path in outputs.values():
        exported = path.read_bytes()
        assert b"/synthetic/private" not in exported
        assert b"image_path" not in exported
