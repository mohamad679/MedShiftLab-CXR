# MedShiftLab-CXR CV Project Summary

## CV-ready bullets

- Built MedShiftLab-CXR, a reproducible chest X-ray evaluation framework for pretrained models under annotation uncertainty and cross-dataset shift.
- Implemented conservative CheXpert/VinDr label harmonization, four explicit uncertainty-label strategies, private-data configuration, image loading, standardized prediction schemas, and auditable JSON/CSV reporting.
- Ran TorchXRayVision DenseNet121 on a 202-image frontal CheXpert validation reference set and on all 15,000 prepared VinDr/VinBigData images without training a new model.
- Computed label-wise AUROC, AUPRC, Brier score, ECE, threshold metrics, calibration-bin summaries, bootstrap summaries, subgroup audits, and a CheXpert-to-VinDr comparison.
- Preserved strict research boundaries: retrospective external-dataset evaluation only, no clinical validation, no diagnostic/deployment claim, and no SOTA claim.

## Short project description

MedShiftLab-CXR is a compute-aware medical-imaging research project that studies how label policy and dataset shift affect a fixed pretrained chest X-ray classifier. The `v1.0.0` release includes a real CheXpert reference evaluation and a complete VinDr/VinBigData external-dataset evaluation chain with full inference on 15,000 images, external metrics, threshold analysis, calibration analysis, and cross-dataset comparison. The repository separates private runtime data from tracked code and sanitized aggregate reports.

## Claim boundary

The completed VinDr/VinBigData work is a retrospective external-dataset evaluation under a conservative four-label mapping. It is not prospective clinical validation, fairness validation, regulatory evidence, deployment validation, or a comprehensive clinical benchmark. The repository has not yet compared separately fitted model heads across `U-ignore`, `U-zero`, `U-one`, and `U-soft`.

## Current version

Package and release version: `1.0.0`.
