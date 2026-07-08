# VinDr/VinBigData Calibration Analysis v0.6.1

## Status

v0.6.1 records calibration-oriented analysis for the VinDr/VinBigData external-validation predictions.

This stage does not train a model and does not run new inference. It reuses the full v0.5.1 prediction file and the prepared VinDr/VinBigData labels.

## Objective

The goal of v0.6.1 was to summarize calibration behavior using 10-bin expected calibration error, mean prediction, observed prevalence, and calibration gap.

## Analysis Setup

| Field | Value |
|---|---:|
| Dataset | VinDr/VinBigData |
| Prediction rows | 15,000 |
| Label rows | 15,000 |
| Calibration bins | 10 |
| Labels analyzed | 4 |

Pneumonia was excluded because VinBigData has zero positive Pneumonia labels under the conservative mapping policy.

## Calibration Summary

| Label | ECE | Mean prediction | Observed prevalence | Calibration gap |
|---|---:|---:|---:|---:|
| Atelectasis | 0.171129 | 0.183529 | 0.012400 | 0.171129 |
| Cardiomegaly | 0.021805 | 0.166982 | 0.153333 | 0.013648 |
| Pleural Effusion | 0.068643 | 0.130785 | 0.068800 | 0.061985 |
| Pneumothorax | 0.208032 | 0.214432 | 0.006400 | 0.208032 |

## Interpretation

Cardiomegaly shows the best calibration among the evaluated labels, with low ECE and a small calibration gap.

Pleural Effusion shows moderate overprediction.

Atelectasis and Pneumothorax show substantial overprediction relative to observed prevalence. This is especially visible for Pneumothorax, where the mean prediction is much higher than the observed positive rate.

These results support calibration or threshold-specific reporting before any deployment-oriented interpretation.

## Boundary

This is calibration analysis from existing external-validation predictions.

It does not establish clinical calibration, deployment readiness, fairness validation, or prospective validation.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, calibration bin CSVs, calibration summary CSVs, summary JSON files, model weights, Kaggle paths, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.6.2 should produce an operating-point recommendation summary combining threshold analysis and calibration findings.
