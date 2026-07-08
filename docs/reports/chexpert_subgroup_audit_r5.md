# CheXpert Safe-Subset R5 Subgroup Audit Report

## Status

This report records an exploratory subgroup/slice audit on the R3 evaluation split.

This is not a full benchmark, not external validation, not fairness validation, and not clinical validation.

## Source run

- Source experiment: R3 calibrated-threshold evaluation
- Runtime: Kaggle Notebook
- Dataset: CheXpert validation split
- Evaluation records: 101
- Model: TorchXRayVision DenseNet121
- Preprocessing version: `phase5-baseline-inference-v2`
- Normalization: `torchxrayvision`
- Subgroup variables:
  - sex
  - view position
  - age bucket

## Subgroup counts

| Subgroup Variable | Value | Records |
|---|---|---:|
| Sex | Male | 55 |
| Sex | Female | 46 |
| View Position | AP | 89 |
| View Position | PA | 12 |
| Age Bucket | 60-79 | 41 |
| Age Bucket | 40-59 | 27 |
| Age Bucket | <40 | 17 |
| Age Bucket | 80+ | 16 |

## Aggregate subgroup metrics

| Subgroup Variable | Value | Threshold Source | Records | Labels with F1 | Mean F1 | Mean Sensitivity | Mean Specificity | Mean Balanced Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Age Bucket | 40-59 | Calibration-selected | 27 | 3 | 0.715737 | 0.480635 | 0.797732 | 0.639183 |
| Age Bucket | 40-59 | Default 0.5 | 27 | 5 | 0.386565 | 1.000000 | 0.000000 | 0.500000 |
| Age Bucket | 60-79 | Calibration-selected | 41 | 3 | 0.537916 | 0.449679 | 0.789211 | 0.596145 |
| Age Bucket | 60-79 | Default 0.5 | 41 | 4 | 0.390762 | 1.000000 | 0.000000 | 0.500000 |
| Age Bucket | 80+ | Calibration-selected | 16 | 3 | 0.478632 | 0.316667 | 0.783333 | 0.550000 |
| Age Bucket | 80+ | Default 0.5 | 16 | 5 | 0.369483 | 1.000000 | 0.000000 | 0.500000 |
| Age Bucket | <40 | Calibration-selected | 17 | 3 | 0.436905 | 0.738095 | 0.774902 | 0.691270 |
| Age Bucket | <40 | Default 0.5 | 17 | 3 | 0.334795 | 1.000000 | 0.000000 | 0.500000 |
| Sex | Female | Calibration-selected | 46 | 3 | 0.593189 | 0.389748 | 0.806114 | 0.597931 |
| Sex | Female | Default 0.5 | 46 | 5 | 0.339503 | 1.000000 | 0.000000 | 0.500000 |
| Sex | Male | Calibration-selected | 55 | 3 | 0.544036 | 0.387368 | 0.778165 | 0.582767 |
| Sex | Male | Default 0.5 | 55 | 5 | 0.315736 | 1.000000 | 0.000000 | 0.500000 |
| View Position | AP | Calibration-selected | 89 | 3 | 0.594305 | 0.418567 | 0.763496 | 0.591032 |
| View Position | AP | Default 0.5 | 89 | 5 | 0.338282 | 1.000000 | 0.000000 | 0.500000 |
| View Position | PA | Calibration-selected | 12 | 1 | 0.500000 | 0.083333 | 0.945455 | 0.507576 |
| View Position | PA | Default 0.5 | 12 | 4 | 0.301923 | 1.000000 | 0.000000 | 0.500000 |

## Interpretation

The subgroup audit confirms the same operating-pattern shift seen in R3 and R4. The default 0.5 threshold produces all-positive behavior across subgroups, with specificity equal to 0.0. Calibration-selected thresholds substantially increase specificity across sex, view-position, and age-bucket slices.

However, this is not a validated subgroup or fairness analysis. Several slices are small, especially PA view position with only 12 evaluation records. Rare labels also have limited positive support, which makes per-slice sensitivity, F1, and balanced accuracy unstable. The `labels with F1` column indicates that some subgroup aggregates are calculated from fewer than all five labels because some label/slice combinations lack sufficient positive or negative support.

The appropriate conclusion is that the subgroup audit infrastructure works and that calibration changes the operating behavior across slices. It should not be used to claim subgroup fairness, clinical validity, or production readiness.

## Reproducibility note

Raw CheXpert images, local paths, sample-level labels, predictions, split files, subgroup CSV files, and Kaggle private outputs were intentionally not committed to the repository.

Only this sanitized summary report and the reusable subgroup audit script are stored in GitHub.
