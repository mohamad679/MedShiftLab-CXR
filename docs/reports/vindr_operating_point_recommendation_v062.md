# VinDr/VinBigData Operating-Point Recommendation v0.6.2

## Status

v0.6.2 closes the VinDr/VinBigData threshold and calibration analysis phase.

This stage does not train a model and does not run new inference. It summarizes operating-point recommendations from the existing full external-validation predictions.

## Input Analyses

| Version | Analysis | Status |
|---|---|---|
| v0.5.2 | External metrics | complete |
| v0.6.0 | Threshold analysis | complete |
| v0.6.1 | Calibration analysis | complete |

## Recommended Reporting Thresholds

| Label | Recommended threshold | Rationale |
|---|---:|---|
| Atelectasis | 0.55 | Best F1 threshold, but performance remains weak |
| Cardiomegaly | 0.35 | Improves F1 over default 0.5 |
| Pleural Effusion | 0.50 | Default threshold already matches best F1 |
| Pneumothorax | 0.50 | Default threshold matches best F1, but precision remains weak |

## Calibration Summary

| Label | ECE | Calibration interpretation |
|---|---:|---|
| Atelectasis | 0.171129 | poorly calibrated / overprediction |
| Cardiomegaly | 0.021805 | best calibrated label |
| Pleural Effusion | 0.068643 | moderate overprediction |
| Pneumothorax | 0.208032 | poorly calibrated / strong overprediction |

## Practical Interpretation

Cardiomegaly and Pleural Effusion are the strongest labels in this VinDr/VinBigData external evaluation.

Atelectasis and Pneumothorax should be treated cautiously because they have low prevalence, weak AUPRC/F1, and poor calibration behavior.

Threshold 0.5 should not be treated as globally optimal. Cardiomegaly benefits from a lower threshold of 0.35 under the F1 objective.

## Boundary

These recommendations are for retrospective external dataset analysis only.

They do not establish clinical calibration, clinical validation, deployment readiness, fairness validation, or prospective validation.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, threshold sweep files, calibration files, metrics files, model weights, Kaggle paths, or private runtime outputs are committed.

Only this sanitized closeout report is tracked.

## Phase Decision

The VinDr/VinBigData operating-point analysis phase is complete.

The next project phase should move to cross-dataset shift analysis, subgroup audit, or final reproducibility packaging.
