# MedShiftLab-CXR Final Release Closeout v1.0.0

## Status

v1.0.0 closes the current MedShiftLab-CXR project scope.

This release documents a complete reproducible evaluation narrative using CheXpert and VinDr/VinBigData with TorchXRayVision DenseNet121.

No new model training, deployment, clinical validation, or additional inference is introduced in this final release.

## Final Scope

MedShiftLab-CXR is a research-oriented chest X-ray dataset-shift evaluation project.

The project demonstrates:

- CheXpert reference evaluation
- VinDr/VinBigData external validation
- real image inference
- real external metrics
- threshold analysis
- calibration analysis
- cross-dataset comparison
- sanitized reporting boundaries

## Final Completed Chain

| Version | Purpose | Status |
|---|---|---|
| v0.1.x | Repository cleanup, metadata loading, reproducibility preparation | complete |
| v0.4.x | VinDr/VinBigData external-validation preparation and schema controls | complete |
| v0.5.0 | VinDr real inference subset | complete |
| v0.5.1 | VinDr full real inference on 15,000 images | complete |
| v0.5.2 | VinDr external metrics | complete |
| v0.5.3 | VinDr external-validation closeout | complete |
| v0.6.0 | VinDr threshold analysis | complete |
| v0.6.1 | VinDr calibration analysis | complete |
| v0.6.2 | VinDr operating-point recommendation closeout | complete |
| v0.7.0 | CheXpert frontal validation reference evaluation | complete |
| v0.7.1 | CheXpert vs VinDr cross-dataset comparison | complete |
| v0.7.2 | CheXpert/VinDr comparison closeout | complete |
| v1.0.0 | Final release closeout | complete |

## Final Key Results

### VinDr/VinBigData External Metrics

| Label | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| Atelectasis | 0.798618 | 0.044507 | 0.086042 |
| Cardiomegaly | 0.887571 | 0.605342 | 0.566805 |
| Pleural Effusion | 0.894115 | 0.606659 | 0.566575 |
| Pneumothorax | 0.823805 | 0.040432 | 0.046148 |

Pneumonia was excluded from VinDr/VinBigData metrics because the conservative mapping produced zero positive Pneumonia labels.

### CheXpert Reference Metrics

| Label | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| Atelectasis | 0.808399 | 0.640529 | 0.583658 |
| Cardiomegaly | 0.808712 | 0.647801 | 0.556522 |
| Pleural Effusion | 0.881907 | 0.772327 | 0.538462 |
| Pneumonia | 0.825387 | 0.169421 | 0.088889 |
| Pneumothorax | 0.639560 | 0.057658 | 0.068293 |

## Final Interpretation

Cardiomegaly and Pleural Effusion are the most stable and interpretable labels across the CheXpert and VinDr/VinBigData evaluations.

Atelectasis and Pneumothorax show weaker precision-sensitive behavior, especially under external evaluation and class imbalance.

Default threshold 0.5 is not a universal operating point. Threshold and calibration analyses are required for responsible interpretation.

The project supports a credible dataset-shift evaluation narrative, not a clinical deployment claim.

## Final Boundaries

This release does not claim:

- clinical validation
- diagnostic deployment readiness
- prospective validation
- fairness validation
- SOTA performance
- full clinical benchmark completion
- FDA/CE readiness
- real-world medical decision support

## Repository Boundary

The repository tracks only code, tests, configs, README updates, and sanitized reports.

The repository does not commit:

- raw images
- private manifests
- private labels
- private predictions
- model weights
- Kaggle runtime outputs
- local private artifacts
- evaluation CSV/JSON outputs from private runs

## Final Release Decision

The current MedShiftLab-CXR scope is complete at v1.0.0.

Future work, if needed, should start as a new post-v1.0 development line rather than extending the current closeout chain.
