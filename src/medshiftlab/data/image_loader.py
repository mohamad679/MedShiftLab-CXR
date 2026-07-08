"""Reusable, model-independent image loading and preprocessing.

This module reads configured local JPEG/PNG files into memory. It does not run
model inference, download data, or copy source images into the repository.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medshiftlab.data.registry import (
    LocalDataConfig,
    get_dataset_registry_entry,
    require_local_dataset_paths,
)


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class UnsupportedImageFormatError(ValueError):
    """Raised when an image extension is outside the supported CXR formats."""


class ImageLoadError(RuntimeError):
    """Raised when a supported image exists but Pillow cannot decode it."""


class ImagePreprocessingConfig(BaseModel):
    """Model-independent image conversion, resize, and normalization settings.

    ``target_size`` is expressed as ``(height, width)``. Standardization first
    scales pixels to ``[0, 1]`` and then applies ``(value - mean) / std``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_mode: Literal["grayscale", "rgb"] = "grayscale"
    target_size: tuple[int, int] | None = None
    normalization: Literal[
        "none", "zero_one", "minus_one_one", "torchxrayvision", "standardize"
    ] = "zero_one"
    standardize_mean: float = 0.0
    standardize_std: float = Field(default=1.0, gt=0.0)

    @field_validator("target_size")
    @classmethod
    def _validate_target_size(
        cls,
        value: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if value is not None and any(dimension <= 0 for dimension in value):
            raise ValueError("target_size dimensions must be positive")
        return value


@dataclass(frozen=True)
class LoadedImage:
    """An in-memory preprocessed image and non-sensitive shape metadata."""

    array: NDArray[np.float32]
    original_mode: str
    original_size: tuple[int, int]
    output_mode: Literal["grayscale", "rgb"]


class ImageLoadIssue(BaseModel):
    """One missing or failed relative image path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_path: str = Field(min_length=1)
    status: Literal["missing", "failed"]
    error: str = Field(min_length=1)


class ImageLoadSummary(BaseModel):
    """Aggregate result from a bounded image-loading audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = Field(min_length=1)
    total: int = Field(ge=0)
    loadable: int = Field(ge=0)
    missing: int = Field(ge=0)
    failed: int = Field(ge=0)
    issues: tuple[ImageLoadIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_counts(self) -> ImageLoadSummary:
        if self.loadable + self.missing + self.failed != self.total:
            raise ValueError("image summary counts must add up to total")
        if len(self.issues) != self.missing + self.failed:
            raise ValueError("image summary issues must describe every missing/failed image")
        return self


def resolve_dataset_image_directory(
    config: LocalDataConfig,
    dataset_name: str,
) -> Path:
    """Resolve the configured image directory without accessing image files."""

    paths = require_local_dataset_paths(config, dataset_name)
    image_directory = paths["image_directory"].expanduser()
    if not image_directory.is_absolute():
        base_directory = (
            config.source_path.parent
            if config.source_path is not None
            else Path.cwd()
        )
        image_directory = base_directory / image_directory
    return image_directory.resolve(strict=False)


def resolve_dataset_image_path(
    config: LocalDataConfig,
    dataset_name: str,
    image_path: str | Path,
) -> Path:
    """Resolve a relative record path beneath the configured image directory."""

    raw_path = str(image_path).strip()
    if not raw_path:
        raise ValueError("image_path must not be blank")

    relative_path = Path(raw_path)
    if relative_path.is_absolute() or PureWindowsPath(raw_path).is_absolute():
        raise ValueError("image_path must be relative to the configured image directory")

    image_directory = resolve_dataset_image_directory(config, dataset_name)
    resolved_path = (image_directory / relative_path).resolve(strict=False)
    if not resolved_path.is_relative_to(image_directory):
        raise ValueError("image_path must remain inside the configured image directory")
    return resolved_path


def load_image(
    path: str | Path,
    preprocessing: ImagePreprocessingConfig | None = None,
) -> LoadedImage:
    """Load and preprocess one local JPEG or PNG image entirely in memory."""

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    extension = image_path.suffix.lower()
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        raise UnsupportedImageFormatError(
            f"Unsupported image format '{extension or '<none>'}' for {image_path.name}. "
            f"Supported formats: {supported}"
        )

    settings = preprocessing or ImagePreprocessingConfig()
    try:
        with Image.open(image_path) as source:
            source = ImageOps.exif_transpose(source)
            original_mode = source.mode
            original_size = source.size
            output = source.convert("L" if settings.output_mode == "grayscale" else "RGB")
            if settings.target_size is not None:
                height, width = settings.target_size
                output = output.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(output, dtype=np.float32).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ImageLoadError(f"Failed to decode image {image_path}: {error}") from error

    array = _normalize_image(array, settings)
    return LoadedImage(
        array=array,
        original_mode=original_mode,
        original_size=original_size,
        output_mode=settings.output_mode,
    )


def load_dataset_image(
    config: LocalDataConfig,
    dataset_name: str,
    image_path: str | Path,
    preprocessing: ImagePreprocessingConfig | None = None,
) -> LoadedImage:
    """Resolve and load one registry-configured dataset image."""

    resolved_path = resolve_dataset_image_path(config, dataset_name, image_path)
    return load_image(resolved_path, preprocessing)


def discover_dataset_image_paths(
    config: LocalDataConfig,
    dataset_name: str,
    *,
    limit: int,
) -> tuple[Path, ...]:
    """Discover at most ``limit`` supported relative paths without following links."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    image_directory = resolve_dataset_image_directory(config, dataset_name)
    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"Configured image directory not found for dataset '{dataset_name}': "
            f"{image_directory}"
        )

    discovered: list[Path] = []
    for directory, child_directories, filenames in os.walk(
        image_directory,
        followlinks=False,
    ):
        child_directories.sort()
        for filename in sorted(filenames):
            candidate = Path(directory) / filename
            if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            discovered.append(candidate.relative_to(image_directory))
            if len(discovered) == limit:
                return tuple(discovered)
    return tuple(discovered)


def summarize_dataset_images(
    config: LocalDataConfig,
    dataset_name: str,
    image_paths: Iterable[str | Path],
    preprocessing: ImagePreprocessingConfig | None = None,
    *,
    limit: int,
) -> ImageLoadSummary:
    """Load a bounded sequence and count loadable, missing, and failed images."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    entry = get_dataset_registry_entry(dataset_name)
    total = 0
    loadable = 0
    missing = 0
    failed = 0
    issues: list[ImageLoadIssue] = []

    for image_path in islice(image_paths, limit):
        total += 1
        display_path = str(image_path)
        try:
            load_dataset_image(config, entry.name, image_path, preprocessing)
        except FileNotFoundError as error:
            missing += 1
            issues.append(
                ImageLoadIssue(
                    image_path=display_path,
                    status="missing",
                    error=_without_resolved_path(error, display_path),
                )
            )
        except (ImageLoadError, UnsupportedImageFormatError, ValueError) as error:
            failed += 1
            issues.append(
                ImageLoadIssue(
                    image_path=display_path,
                    status="failed",
                    error=_without_resolved_path(error, display_path),
                )
            )
        else:
            loadable += 1

    return ImageLoadSummary(
        dataset_name=entry.name,
        total=total,
        loadable=loadable,
        missing=missing,
        failed=failed,
        issues=tuple(issues),
    )


def _normalize_image(
    array: NDArray[np.float32],
    settings: ImagePreprocessingConfig,
) -> NDArray[np.float32]:
    if settings.normalization == "none":
        normalized = array
    elif settings.normalization == "zero_one":
        normalized = array / 255.0
    elif settings.normalization == "minus_one_one":
        normalized = (array / 127.5) - 1.0
    elif settings.normalization == "torchxrayvision":
        normalized = ((array / 255.0) * 2.0 - 1.0) * 1024.0
    else:
        normalized = (
            (array / 255.0) - settings.standardize_mean
        ) / settings.standardize_std
    return normalized.astype(np.float32, copy=False)


def _without_resolved_path(error: Exception, display_path: str) -> str:
    """Return a summary-safe error without exposing a resolved private path."""

    if isinstance(error, FileNotFoundError):
        return f"Image file not found: {display_path}"
    if isinstance(error, UnsupportedImageFormatError):
        extension = Path(display_path).suffix.lower() or "<none>"
        return f"Unsupported image format: {extension}"
    if isinstance(error, ImageLoadError):
        return "Image could not be decoded"
    return str(error)
