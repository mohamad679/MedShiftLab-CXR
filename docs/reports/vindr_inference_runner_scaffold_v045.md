# VinDr/VinBigData Inference Runner Scaffold v0.4.5

## Status

v0.4.5 adds a VinDr/VinBigData inference-runner scaffold.

This stage is not real model inference, not model training, and not external-validation metrics.

## Objective

The goal of v0.4.5 is to define a safe CLI boundary for future VinDr/VinBigData inference runs.

The scaffold validates prepared manifest and label inputs and writes a local/private inference-readiness summary.

## Added Components

| Component | Purpose |
|---|---|
| `scripts/run_vindr_inference_scaffold.py` | Validates manifest and labels before future inference |
| `tests/test_medshiftlab_vindr_inference_scaffold.py` | Covers scaffold success and fail-fast behavior |

## Safety Checks

The scaffold fails when:

- the manifest or labels CSV is missing
- the manifest or labels CSV is empty
- required columns are missing
- no images are marked as found
- any manifest image is marked as missing
- manifest image paths are absolute instead of relative
- labels and manifest sample IDs do not match

## Output Boundary

The scaffold writes only a summary JSON.

It does not generate predictions and does not compute metrics.

## Claim Boundary

The scaffold summary explicitly records:

| Claim | Value |
|---|---|
| model loaded | false |
| inference completed | false |
| external validation completed | false |
| clinical validation completed | false |
| metrics completed | false |

## Repository Boundary

No raw images, local manifests, local labels, local summaries, prediction files, Kaggle paths, or private runtime outputs are committed.

Only sanitized code, tests, README text, and this report are tracked.

## Next Stage

v0.4.6 may connect the scaffold to a real model adapter or define a controlled dry-run interface for prediction generation.

Metrics must remain separate until prediction files are actually generated and evaluated.
