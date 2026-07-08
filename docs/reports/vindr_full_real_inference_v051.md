# VinDr/VinBigData Full Real Inference v0.5.1

## Status

v0.5.1 records the first full real VinDr/VinBigData image inference run.

This stage runs real model inference on the full prepared VinDr/VinBigData image set. It does not compute external-validation metrics.

## Objective

The goal of v0.5.1 was to verify that the repository inference path can run a pretrained TorchXRayVision DenseNet model across all prepared VinDr/VinBigData images and produce complete project-label prediction outputs.

## Inference Result

| Check | Result |
|---|---:|
| Model family | TorchXRayVision DenseNet121 |
| Weights | densenet121-res224-all |
| Device | cuda |
| Full VinDr images inferred | 15,000 |
| Batch/chunk count | 469 |
| Project prediction rows | 15,000 |
| Project target labels | 5 |
| NaN values | 0 |
| Minimum mapped prediction | 0.000005044072 |
| Maximum mapped prediction | 0.9772801 |

## Project Prediction Columns

| Column |
|---|
| sample_id |
| Atelectasis |
| Cardiomegaly |
| Pleural Effusion |
| Pneumonia |
| Pneumothorax |

## Prediction Range by Label

| Label | Min | Max |
|---|---:|---:|
| Atelectasis | 0.00036559283 | 0.65950036 |
| Cardiomegaly | 0.00007577255 | 0.9651904 |
| Pleural Effusion | 0.0021183628 | 0.9772801 |
| Pneumonia | 0.000005044072 | 0.9670335 |
| Pneumothorax | 0.0032477407 | 0.5417419 |

## Schema Validation

The mapped full project prediction file passed prediction-schema validation with exact sample coverage.

| Check | Result |
|---|---:|
| Prediction rows | 15,000 |
| Manifest rows | 15,000 |
| Label rows | 15,000 |
| Manifest-label overlap | 15,000 |
| Exact sample coverage required | yes |
| Metrics generated | no |

## Boundary

This stage confirms full real inference execution and prediction-schema compatibility.

It does not compute AUROC, AUPRC, F1, sensitivity, specificity, calibration, subgroup metrics, fairness metrics, or clinical validation.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, metrics files, Kaggle paths, model weights, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.5.2 should compute external-validation metrics from the full prediction file and prepared labels.
