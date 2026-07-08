# VinDr/VinBigData Inference-Readiness Handoff v0.4.4

## Status

v0.4.4 records the inference-readiness handoff for VinDr/VinBigData external validation.

This stage is not model training, not model inference, and not external-validation metrics.

## Objective

The goal of v0.4.4 is to close the input-preparation phase and define the minimum safe handoff requirements before any VinDr/VinBigData image inference is attempted.

## Completed Preconditions

The previous stages established the following:

| Stage | Result |
|---|---|
| v0.4.0 | VinDr external-validation scaffold added |
| v0.4.1 | Real metadata dry-run completed |
| v0.4.2 | Reusable input-preparation workflow hardened |
| v0.4.3 | Real Kaggle input workflow run completed |

## Real Input Workflow Result

The v0.4.3 Kaggle run verified:

| Check | Result |
|---|---:|
| Unique annotation image IDs | 15,000 |
| Manifest rows | 15,000 |
| Images found | 15,000 |
| Images missing | 0 |
| Require images enabled | yes |

## Conservative Label Set

The current VinDr/VinBigData input workflow prepares the following project labels:

| Label | Status |
|---|---|
| Atelectasis | mapped |
| Cardiomegaly | mapped |
| Pleural Effusion | mapped |
| Pneumonia | retained as project target but zero-positive for VinBigData |
| Pneumothorax | mapped |

Pneumonia remains zero-positive because VinBigData does not provide a direct Pneumonia class under the conservative mapping policy.

## Inference-Readiness Requirements

Before running inference, the project must have:

1. A fixed model adapter and checkpoint/source definition.
2. A reproducible image loading path for the prepared VinDr manifest.
3. A prediction output schema that is clearly separated from labels and metadata.
4. A local/private output location for prediction files.
5. Fail-fast checks for missing images, empty manifests, and empty predictions.
6. Documentation stating that inference outputs are not clinical validation.

## Repository Boundary

No raw images, local manifests, local labels, local summaries, prediction files, Kaggle paths, or private runtime outputs are committed.

Only this sanitized handoff report is tracked.

## Next Stage

v0.4.5 should add the VinDr inference runner scaffold.

That next stage may define a prediction-file schema and CLI interface, but should still avoid claiming external-validation metrics until predictions are actually generated and evaluated.
