# VinDr/VinBigData External Validation Closeout v0.5.3

## Status

v0.5.3 closes the VinDr/VinBigData external-validation chain for MedShiftLab-CXR.

This closeout summarizes the completed real-data preparation, real inference, and real external metric computation stages.

## Completed Chain

| Version | Purpose | Status |
|---|---|---|
| v0.4.0 | VinDr external-validation scaffold | complete |
| v0.4.1 | Real metadata dry-run | complete |
| v0.4.2 | Reusable input workflow hardening | complete |
| v0.4.3 | Real input workflow run | complete |
| v0.4.4 | Inference-readiness handoff | complete |
| v0.4.5 | Inference runner scaffold | complete |
| v0.4.6 | Real inference scaffold run | complete |
| v0.4.7 | Prediction schema contract | complete |
| v0.4.8 | Prediction schema validation run | complete |
| v0.5.0 | Real inference on 25-image subset | complete |
| v0.5.1 | Full real inference on 15,000 images | complete |
| v0.5.2 | External metric computation | complete |

## Final Evaluation Setup

| Field | Value |
|---|---:|
| Dataset | VinDr/VinBigData |
| Images evaluated | 15,000 |
| Model | TorchXRayVision DenseNet121 |
| Weights | densenet121-res224-all |
| Prediction rows | 15,000 |
| Label rows | 15,000 |
| Manifest rows | 15,000 |
| Threshold | 0.5 |
| Calibration bins | 10 |
| Metrics labels | 4 |

Pneumonia was excluded from metrics because VinBigData has zero positive Pneumonia labels under the conservative mapping policy.

## Final External Metrics

| Label | Positives | AUROC | AUPRC | F1 | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 186 | 0.798618 | 0.044507 | 0.086042 | 0.483871 | 0.877413 | 0.066375 | 0.171129 |
| Cardiomegaly | 2,300 | 0.887571 | 0.605342 | 0.566805 | 0.563478 | 0.923071 | 0.089504 | 0.021805 |
| Pleural Effusion | 1,032 | 0.894115 | 0.606659 | 0.566575 | 0.597868 | 0.962128 | 0.049629 | 0.068643 |
| Pneumothorax | 96 | 0.823805 | 0.040432 | 0.046148 | 0.645833 | 0.830314 | 0.076939 | 0.208032 |

## Key Findings

The pretrained TorchXRayVision DenseNet121 model achieved strong ranking performance for Cardiomegaly and Pleural Effusion on VinDr/VinBigData, with AUROC above 0.88 and AUPRC above 0.60 for both labels.

Atelectasis and Pneumothorax had acceptable AUROC but weak AUPRC and F1, consistent with severe label imbalance and threshold sensitivity.

The threshold 0.5 operating point is not optimized and should not be interpreted as clinically calibrated.

## Boundary

This is external dataset evaluation, not clinical validation.

This closeout does not claim:

- diagnostic deployment readiness
- clinical validation
- fairness validation
- prospective validation
- SOTA performance
- full benchmark completion

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, standardized prediction JSON, metrics JSON, metrics CSV, model weights, Kaggle paths, or private runtime outputs are committed.

Only sanitized reports, code, tests, and README updates are tracked.

## VinDr Phase Decision

The VinDr/VinBigData external-validation phase is complete for the current project scope.

Future work should move to cross-dataset shift analysis, calibration, subgroup audit, or final reproducibility packaging.
