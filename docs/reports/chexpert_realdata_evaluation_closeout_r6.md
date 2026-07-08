# CheXpert Real-Data Evaluation Closeout R6

## Status

This document closes out the first real-data CheXpert evaluation sequence for MedShiftLab-CXR.

The sequence covers R1 through R5:

- R1: real-image CheXpert safe-subset inference/evaluation
- R2: threshold sweep
- R3: calibrated threshold evaluation split
- R4: bootstrap uncertainty
- R5: subgroup/slice audit

This is not a full benchmark, not external validation, not fairness validation, and not clinical validation.

## Version chain

| Tag | Scope |
|---|---|
| `v0.3.0-roadmap-closeout` | Roadmap/documentation closeout |
| `v0.3.1-chexpert-safe-subset-r1` | First real CheXpert safe-subset inference/evaluation |
| `v0.3.2-chexpert-threshold-sweep-r2` | Threshold sweep |
| `v0.3.3-chexpert-calibrated-threshold-eval-r3` | Calibration/evaluation split |
| `v0.3.4-chexpert-bootstrap-uncertainty-r4` | Bootstrap uncertainty |
| `v0.3.5-chexpert-subgroup-audit-r5` | Subgroup/slice audit |

## Core technical fix

The real-data run exposed and fixed the TorchXRayVision preprocessing scale issue.

- Previous behavior: `minus_one_one`
- Corrected behavior: `torchxrayvision`
- Corrected preprocessing version: `phase5-baseline-inference-v2`
- Expected TorchXRayVision input scale: approximately `[-1024, 1024]`

## R1 summary

A local/private Kaggle run completed on 64 frontal CheXpert validation images.

Aggregate metrics at threshold 0.5:

| Metric | Value |
|---|---:|
| Mean AUROC | 0.7999289127474138 |
| Mean AUPRC | 0.43391043300391524 |
| Mean F1 | 0.23849916690252998 |
| Mean Sensitivity | 1.0 |
| Mean Specificity | 0.0 |

Interpretation: ranking behavior was useful on the small exploratory subset, but the default threshold produced all-positive behavior.

## R2 summary

A per-label threshold sweep on the same 64-image subset found exploratory thresholds between 0.63 and 0.69.

| Label | Threshold |
|---|---:|
| Atelectasis | 0.68 |
| Cardiomegaly | 0.63 |
| Pleural Effusion | 0.69 |
| Pneumonia | 0.65 |
| Pneumothorax | 0.64 |

Interpretation: threshold tuning improved operating behavior on the same subset, but this was not a validated threshold protocol.

## R3 summary

A deterministic calibration/evaluation split was run on 202 frontal CheXpert validation images.

- Calibration records: 101
- Evaluation records: 101

Evaluation aggregate comparison:

| Threshold Source | Mean F1 | Mean Sensitivity | Mean Specificity | Mean Balanced Accuracy |
|---|---:|---:|---:|---:|
| Default 0.5 | 0.3287709977784775 | 1.0 | 0.0 | 0.5 |
| Calibration-selected | 0.344235351836277 | 0.39243447993448 | 0.7873732236058923 | 0.5899038517701862 |

Interpretation: calibration-selected thresholds avoided all-positive behavior and improved specificity/balanced accuracy, while sensitivity dropped, especially for rare labels.

## R4 summary

A bootstrap uncertainty summary was run on the R3 evaluation split with 1000 bootstrap resamples.

| Threshold Source | Metric | Point Estimate | 2.5% | 97.5% |
|---|---|---:|---:|---:|
| Calibration-selected | Mean F1 | 0.344235 | 0.297900 | 0.380221 |
| Calibration-selected | Mean Specificity | 0.787373 | 0.730326 | 0.838532 |
| Calibration-selected | Mean Balanced Accuracy | 0.589904 | 0.556959 | 0.621668 |
| Default 0.5 | Mean F1 | 0.328771 | 0.293198 | 0.360775 |
| Default 0.5 | Mean Specificity | 0.000000 | 0.000000 | 0.000000 |
| Default 0.5 | Mean Balanced Accuracy | 0.500000 | 0.400000 | 0.500000 |

Interpretation: specificity and balanced accuracy clearly changed after calibration, but F1 uncertainty remained close/overlapping.

## R5 summary

A subgroup/slice audit was run on the R3 evaluation split.

Subgroup variables:

- Sex
- View position
- Age bucket

Main finding: default 0.5 retained all-positive behavior across slices, while calibration-selected thresholds increased specificity across evaluated subgroups.

Important limitation: subgroup sizes were small in some slices, especially PA view position with 12 records. Rare-label estimates remain unstable.

## Repository outputs

The following reusable scripts were added:

- `scripts/run_threshold_sweep.py`
- `scripts/run_calibrated_threshold_evaluation.py`
- `scripts/run_bootstrap_uncertainty.py`
- `scripts/run_subgroup_audit.py`

The following sanitized reports were added:

- `docs/reports/chexpert_safe_subset_r1.md`
- `docs/reports/chexpert_threshold_sweep_r2.md`
- `docs/reports/chexpert_calibrated_threshold_eval_r3.md`
- `docs/reports/chexpert_bootstrap_uncertainty_r4.md`
- `docs/reports/chexpert_subgroup_audit_r5.md`

## Data handling

Raw CheXpert images, local paths, sample-level labels, prediction files, split files, bootstrap CSV files, subgroup CSV files, and Kaggle private outputs were intentionally not committed to the repository.

Only sanitized reports and reusable evaluation scripts were committed.

## Final interpretation

MedShiftLab-CXR now has a working real-image evaluation path with threshold calibration, uncertainty summaries, and subgroup audit tooling.

The current evidence supports pipeline functionality and exploratory operating-behavior analysis. It does not support clinical, diagnostic, deployment, external-validation, fairness-validation, or full-benchmark claims.

## Recommended next phase

The next meaningful phase should be one of:

1. External dataset validation on MIMIC-CXR-JPG or VinDr-CXR.
2. Larger CheXpert validation run with a stricter calibration/evaluation protocol.
3. Packaging the evaluation workflow into a reproducible command sequence or notebook.
