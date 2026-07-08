# VinDr/VinBigData Threshold Analysis v0.6.0

## Status

v0.6.0 records threshold tuning and operating-point analysis for the VinDr/VinBigData external-validation predictions.

This stage does not train a model and does not run new inference. It reuses the full v0.5.1 prediction file and the prepared VinDr/VinBigData labels.

## Objective

The goal of v0.6.0 was to evaluate whether the default threshold 0.5 is a reasonable operating point for each label and to identify better per-label thresholds for F1 and Youden J.

## Analysis Setup

| Field | Value |
|---|---:|
| Dataset | VinDr/VinBigData |
| Prediction rows | 15,000 |
| Label rows | 15,000 |
| Threshold grid | 0.05 to 0.95 |
| Step size | 0.05 |
| Labels analyzed | 4 |

Pneumonia was excluded because VinBigData has zero positive Pneumonia labels under the conservative mapping policy.

## Best F1 Thresholds

| Label | Best F1 threshold | Best F1 | Precision | Sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.55 | 0.086629 | 0.066667 | 0.123656 | 0.978264 | 23 | 322 | 14,492 | 163 |
| Cardiomegaly | 0.35 | 0.579401 | 0.508882 | 0.672609 | 0.882441 | 1,547 | 1,493 | 11,207 | 753 |
| Pleural Effusion | 0.50 | 0.566575 | 0.538394 | 0.597868 | 0.962128 | 617 | 529 | 13,439 | 415 |
| Pneumothorax | 0.50 | 0.046148 | 0.023929 | 0.645833 | 0.830314 | 62 | 2,529 | 12,375 | 34 |

## Best Youden J Thresholds

| Label | Best Youden threshold | Youden J | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Atelectasis | 0.20 | 0.459622 | 0.790323 | 0.669299 |
| Cardiomegaly | 0.15 | 0.608706 | 0.861304 | 0.747402 |
| Pleural Effusion | 0.30 | 0.607177 | 0.695736 | 0.911440 |
| Pneumothorax | 0.40 | 0.524473 | 0.739583 | 0.784890 |

## Interpretation

The default 0.5 threshold is reasonable for Pleural Effusion and Pneumothorax under the F1 objective, but not for Cardiomegaly, where 0.35 improves F1.

Atelectasis remains weak under F1 even after threshold tuning, likely due to low prevalence and poor precision.

Youden J thresholds favor sensitivity/specificity trade-offs and are generally lower than F1-optimal thresholds.

## Boundary

This is operating-point analysis from existing external-validation predictions.

It does not establish clinical calibration, deployment readiness, fairness validation, or prospective validation.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, threshold sweep CSVs, summary JSON files, model weights, Kaggle paths, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.6.1 should perform calibration-oriented analysis or create a final operating-point recommendation report.
