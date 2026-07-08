# VinDr/VinBigData External Metrics v0.5.2

## Status

v0.5.2 records real external-validation metrics from the full VinDr/VinBigData inference run.

This stage computes metrics from real model predictions and real VinDr/VinBigData labels. It is not clinical validation and not a full benchmark.

## Objective

The goal of v0.5.2 was to evaluate the full v0.5.1 TorchXRayVision prediction file against prepared VinDr/VinBigData labels.

## Evaluation Setup

| Field | Value |
|---|---:|
| Dataset | VinDr/VinBigData |
| Model | TorchXRayVision DenseNet121 |
| Weights | densenet121-res224-all |
| Evaluated records | 15,000 |
| Skipped records | 0 |
| Threshold | 0.5 |
| Calibration bins | 10 |
| Prediction format | standardized JSON |
| Metrics rows | 4 |

Pneumonia was excluded from metric computation because VinBigData has zero positive Pneumonia labels under the conservative mapping policy.

## Label Metrics

| Label | Positives | Negatives | AUROC | AUPRC | F1 | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 186 | 14,814 | 0.798618 | 0.044507 | 0.086042 | 0.483871 | 0.877413 | 0.066375 | 0.171129 |
| Cardiomegaly | 2,300 | 12,700 | 0.887571 | 0.605342 | 0.566805 | 0.563478 | 0.923071 | 0.089504 | 0.021805 |
| Pleural Effusion | 1,032 | 13,968 | 0.894115 | 0.606659 | 0.566575 | 0.597868 | 0.962128 | 0.049629 | 0.068643 |
| Pneumothorax | 96 | 14,904 | 0.823805 | 0.040432 | 0.046148 | 0.645833 | 0.830314 | 0.076939 | 0.208032 |

## Confusion Counts at Threshold 0.5

| Label | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Atelectasis | 90 | 1,816 | 12,998 | 96 |
| Cardiomegaly | 1,296 | 977 | 11,723 | 1,004 |
| Pleural Effusion | 617 | 529 | 13,439 | 415 |
| Pneumothorax | 62 | 2,529 | 12,375 | 34 |

## Interpretation Boundary

These are external dataset metrics for a pretrained model on VinDr/VinBigData labels.

They do not establish clinical validation, deployment readiness, fairness validation, or a complete benchmark.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, standardized prediction JSON, metrics JSON, metrics CSV, model weights, Kaggle paths, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.5.3 should create the final VinDr/VinBigData external-validation closeout report and summarize the v0.4.x to v0.5.x chain.
