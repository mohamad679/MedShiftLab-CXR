"""Synthetic-only tests for reusable image loading and preprocessing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medshiftlab.data import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImagePreprocessingConfig,
    UnsupportedImageFormatError,
    load_dataset_image,
    load_image,
    load_local_data_config,
    resolve_dataset_image_path,
    summarize_dataset_images,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "data" / "example_local_paths.yaml"


def _local_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text(
        """version: 1
datasets:
  chexpert:
    root_path: .
    metadata_path: metadata.csv
    image_directory: images
""",
        encoding="utf-8",
    )
    return config_path


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(("src", "."))
    return environment


def test_load_grayscale_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("L", (4, 3), 128).save(path)

    loaded = load_image(
        path,
        ImagePreprocessingConfig(normalization="none"),
    )

    assert loaded.original_mode == "L"
    assert loaded.original_size == (4, 3)
    assert loaded.output_mode == "grayscale"
    assert loaded.array.shape == (3, 4)
    assert loaded.array.dtype == np.float32
    assert np.all(loaded.array == 128.0)


def test_convert_grayscale_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    Image.new("L", (4, 3), 64).save(path)

    loaded = load_image(
        path,
        ImagePreprocessingConfig(output_mode="rgb", normalization="none"),
    )

    assert loaded.array.shape == (3, 4, 3)
    np.testing.assert_array_equal(loaded.array[..., 0], loaded.array[..., 1])
    np.testing.assert_array_equal(loaded.array[..., 1], loaded.array[..., 2])


def test_resize_uses_height_width_order(tmp_path: Path) -> None:
    path = tmp_path / "image.jpeg"
    Image.new("L", (8, 6), 32).save(path)

    loaded = load_image(
        path,
        ImagePreprocessingConfig(target_size=(5, 7), normalization="none"),
    )

    assert loaded.original_size == (8, 6)
    assert loaded.array.shape == (5, 7)


def test_normalization_modes(tmp_path: Path) -> None:
    path = tmp_path / "values.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L").save(path)

    zero_one = load_image(
        path,
        ImagePreprocessingConfig(normalization="zero_one"),
    ).array
    minus_one_one = load_image(
        path,
        ImagePreprocessingConfig(normalization="minus_one_one"),
    ).array
    standardized = load_image(
        path,
        ImagePreprocessingConfig(
            normalization="standardize",
            standardize_mean=0.5,
            standardize_std=0.5,
        ),
    ).array
    torchxrayvision = load_image(
        path,
        ImagePreprocessingConfig(normalization="torchxrayvision"),
    ).array

    np.testing.assert_allclose(zero_one, [[0.0, 1.0]])
    np.testing.assert_allclose(minus_one_one, [[-1.0, 1.0]])
    np.testing.assert_allclose(standardized, [[-1.0, 1.0]])
    np.testing.assert_allclose(torchxrayvision, [[-1024.0, 1024.0]])


def test_missing_image_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(tmp_path / "missing.png")


def test_unsupported_image_format_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "image.bmp"
    Image.new("L", (2, 2)).save(path)

    with pytest.raises(UnsupportedImageFormatError, match="Unsupported image format"):
        load_image(path)


def test_supported_extensions_are_jpeg_and_png_only() -> None:
    assert SUPPORTED_IMAGE_EXTENSIONS == (".jpg", ".jpeg", ".png")


def test_registry_path_resolution_and_loading(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    Image.new("L", (3, 2), 255).save(image_directory / "one.png")
    config = load_local_data_config(_local_config(tmp_path))

    resolved = resolve_dataset_image_path(config, "chexpert", "one.png")
    loaded = load_dataset_image(config, "chexpert", "one.png")

    assert resolved == image_directory / "one.png"
    assert loaded.array.shape == (2, 3)
    assert np.all(loaded.array == 1.0)


def test_registry_path_resolution_rejects_escape(tmp_path: Path) -> None:
    config = load_local_data_config(_local_config(tmp_path))

    with pytest.raises(ValueError, match="remain inside"):
        resolve_dataset_image_path(config, "chexpert", "../outside.png")


def test_summary_counts_loadable_missing_and_failed_images(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    Image.new("L", (3, 2), 255).save(image_directory / "good.png")
    Image.new("L", (3, 2), 0).save(image_directory / "unsupported.bmp")
    (image_directory / "corrupt.png").write_text("not an image", encoding="utf-8")
    config = load_local_data_config(_local_config(tmp_path))

    summary = summarize_dataset_images(
        config,
        "chexpert",
        ("good.png", "missing.png", "unsupported.bmp", "corrupt.png"),
        limit=4,
    )

    assert summary.total == 4
    assert summary.loadable == 1
    assert summary.missing == 1
    assert summary.failed == 2
    assert {issue.image_path for issue in summary.issues} == {
        "missing.png",
        "unsupported.bmp",
        "corrupt.png",
    }
    assert all(str(tmp_path) not in issue.error for issue in summary.issues)


def test_no_raw_test_images_are_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".dcm",
        ".dicom",
        ".nii",
        ".gz",
        ".zip",
    }

    assert all(Path(path).suffix.lower() not in forbidden_suffixes for path in result.stdout.splitlines())


def test_smoke_script_fails_clearly_without_local_dataset_paths() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_image_loading_smoke_test.py",
            "--config",
            str(EXAMPLE_CONFIG),
            "--dataset",
            "chexpert",
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Missing required local path" in result.stderr


def test_smoke_script_uses_only_temporary_synthetic_images(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    Image.new("L", (4, 3), 128).save(image_directory / "one.png")
    Image.new("L", (4, 3), 64).save(image_directory / "two.png")
    config_path = _local_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_image_loading_smoke_test.py",
            "--config",
            str(config_path),
            "--dataset",
            "chexpert",
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["total"] == 1
    assert summary["loadable"] == 1
    assert summary["missing"] == 0
    assert summary["failed"] == 0
    assert sorted(path.name for path in image_directory.iterdir()) == ["one.png", "two.png"]
