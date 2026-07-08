# VinDr/VinBigData Real Inference Scaffold Run v0.4.6

## Status

v0.4.6 records a real Kaggle execution of the VinDr/VinBigData inference-runner scaffold.

This stage is not model training, not real model inference, and not external-validation metrics.

## Objective

The goal of v0.4.6 was to run the official inference scaffold against real prepared VinDr/VinBigData manifest and label inputs.

The run verifies that the future inference boundary can consume real prepared inputs safely before any model execution is added.

## Execution Result

| Check | Result |
|---|---:|
| Repository commit used | 2a167c7 |
| Scaffold completed | yes |
| Manifest rows | 15,000 |
| Images found | 15,000 |
| Images missing | 0 |
| Label rows | 15,000 |
| Joinable samples | 15,000 |

## Label Counts

| Label | Positive count |
|---|---:|
| Atelectasis | 186 |
| Cardiomegaly | 2,300 |
| Pleural Effusion | 1,032 |
| Pneumonia | 0 |
| Pneumothorax | 96 |

Pneumonia remains zero-positive because VinBigData does not provide a direct Pneumonia class under the conservative mapping policy.

## Claim Boundary

The scaffold summary explicitly records:

| Claim | Value |
|---|---|
| Model loaded | false |
| Inference completed | false |
| External validation completed | false |
| Clinical validation completed | false |
| Metrics completed | false |

## Output Boundary

The run produced a local/private scaffold summary only.

It did not generate prediction files and did not generate metrics files.

## Repository Boundary

No raw images, local manifests, local labels, local summaries, prediction files, Kaggle paths, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.4.7 should define a controlled prediction-generation interface or model-adapter handoff.

Actual external-validation metrics must remain separate until real predictions exist.
