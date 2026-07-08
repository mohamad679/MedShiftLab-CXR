# VinDr/VinBigData External-Validation Input Workflow v0.4.2

## Status

v0.4.2 hardens the reusable VinDr/VinBigData external-validation input workflow for real-data preparation.

This stage is not model training, not inference, and not external-validation metrics.

## Objective

The goal of v0.4.2 is to make local/private VinDr or VinBigData input preparation fail fast when mounted image inputs are incomplete or misconfigured.

## Changes

The `scripts/prepare_vindr_external_validation_inputs.py` scaffold now adds:

- `--require-images`
- `--min-images-found` with default `1`
- explicit `--image-root` existence and directory validation
- explicit failure when the prepared manifest is empty
- explicit failure when no images are found under the provided image root
- explicit failure when required images are missing

Manifest image paths remain relative to the provided image root so local/private mounts are not expanded into absolute image-path entries.

## Validation Boundary

The workflow prepares labels, optional metadata, optional manifests, and a local/private summary only.

It does not:

- train models
- run inference
- compute external-validation metrics
- establish clinical validation
- establish fairness validation
- establish a completed benchmark

## Test Coverage

The scaffold tests now cover:

- successful manifest preparation when matching images are present
- failure when the image root exists but no prepared images match
- failure when the image root path does not exist
- failure when `--require-images` is enabled and some prepared images are missing

## Privacy Boundary

No raw images, manifests, labels, metadata, predictions, or local/private run outputs are included in Git for this stage.

Only this sanitized report, the hardened script, the README update, and the tests are tracked.
