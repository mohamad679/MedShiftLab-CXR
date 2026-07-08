# CheXpert Safe-Subset R4 Bootstrap Uncertainty Report

## Status

This report records an exploratory bootstrap uncertainty summary on the R3 evaluation split.

This is not a full benchmark, not external validation, and not clinical validation. The intervals below are exploratory bootstrap summaries, not formal clinical confidence intervals.

## Source run

- Source experiment: R3 calibrated-threshold evaluation
- Runtime: Kaggle Notebook
- Dataset: CheXpert validation split
- Evaluation records: 101
- Model: TorchXRayVision DenseNet121
- Preprocessing version: `phase5-baseline-inference-v2`
- Normalization: `torchxrayvision`
- Bootstrap resamples: 1000
- Bootstrap seed: 20260708

## Bootstrap summary

| Threshold Source | Metric | Point Estimate | Bootstrap Mean | 2.5% | Median | 97.5% |
|---|---|---:|---:|---:|---:|---:|
| Calibration-selected | Mean F1 | 0.344235 | 0.341205 | 0.297900 | 0.341844 | 0.380221 |
| Calibration-selected | Mean Sensitivity | 0.392434 | 0.391791 | 0.327280 | 0.390747 | 0.454386 |
| Calibration-selected | Mean Specificity | 0.787373 | 0.787351 | 0.730326 | 0.788653 | 0.838532 |
| Calibration-selected | Mean Balanced Accuracy | 0.589904 | 0.589571 | 0.556959 | 0.589851 | 0.621668 |
| Default 0.5 | Mean F1 | 0.328771 | 0.327236 | 0.293198 | 0.326924 | 0.360775 |
| Default 0.5 | Mean Sensitivity | 1.000000 | 0.988200 | 0.800000 | 1.000000 | 1.000000 |
| Default 0.5 | Mean Specificity | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Default 0.5 | Mean Balanced Accuracy | 0.500000 | 0.494100 | 0.400000 | 0.500000 | 0.500000 |

## Interpretation

The bootstrap summary confirms the R3 finding that calibration-selected thresholds avoid the all-positive behavior of the default 0.5 threshold. The calibration-selected thresholds show substantially higher specificity and higher balanced accuracy on the evaluation split.

The mean F1 intervals are close and partially overlapping, so this experiment should not be interpreted as a strong claim of superior F1 performance. The main supported finding is that calibrated thresholds change the operating behavior by trading sensitivity for specificity.

Rare labels remain unstable because Pneumonia and Pneumothorax have very low positive support. This experiment remains exploratory and should not be used as a clinical or production operating point.

## Reproducibility note

Raw CheXpert images, local paths, sample-level labels, predictions, split files, bootstrap CSV files, and Kaggle private outputs were intentionally not committed to the repository.

Only this sanitized summary report and the reusable bootstrap uncertainty script are stored in GitHub.
