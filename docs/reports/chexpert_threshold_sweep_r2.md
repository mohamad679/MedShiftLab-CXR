# CheXpert Safe-Subset R2 Threshold Sweep Report

## Status

This report records an exploratory per-label threshold sweep on the same local/private 64-image frontal CheXpert validation subset used in R1.

This is not a full benchmark, not external validation, and not clinical validation.

## Source run

- Source run: CheXpert safe-subset R1
- Runtime: Kaggle Notebook
- Dataset: CheXpert validation split
- Image subset: 64 frontal validation images
- Model: TorchXRayVision DenseNet121
- Preprocessing version: `phase5-baseline-inference-v2`
- Normalization: `torchxrayvision`
- Threshold grid: 0.00 to 1.00 with step 0.01
- Selection rule:
  1. maximize F1
  2. maximize balanced accuracy
  3. maximize specificity
  4. minimize distance to 0.5
  5. prefer lower threshold

## R1 default-threshold baseline

At threshold 0.5, the R1 aggregate metrics were:

| Metric | Value |
|---|---:|
| Mean AUROC | 0.7999289127474138 |
| Mean AUPRC | 0.43391043300391524 |
| Mean F1 | 0.23849916690252998 |
| Mean Sensitivity | 1.0 |
| Mean Specificity | 0.0 |

The default threshold produced all-positive threshold behavior on this subset.

## R2 best per-label thresholds

| Label | Threshold | F1 | Precision | Sensitivity | Specificity | Balanced Accuracy | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.68 | 0.666667 | 0.611111 | 0.733333 | 0.857143 | 0.795238 | 11 | 7 | 42 | 4 |
| Cardiomegaly | 0.63 | 0.571429 | 0.428571 | 0.857143 | 0.680000 | 0.768571 | 12 | 16 | 34 | 2 |
| Pleural Effusion | 0.69 | 0.666667 | 0.583333 | 0.777778 | 0.909091 | 0.843434 | 7 | 5 | 50 | 2 |
| Pneumonia | 0.65 | 0.347826 | 0.210526 | 1.000000 | 0.750000 | 0.875000 | 4 | 15 | 45 | 0 |
| Pneumothorax | 0.64 | 0.181818 | 0.125000 | 0.333333 | 0.885246 | 0.609290 | 1 | 7 | 54 | 2 |

## Interpretation

The threshold sweep confirms that the default threshold of 0.5 is not appropriate for this small CheXpert safe subset. Per-label thresholds between 0.63 and 0.69 substantially improve specificity and F1 compared with the default all-positive behavior.

However, these thresholds were selected and evaluated on the same 64-image exploratory subset. They are therefore not validated operating points and should not be used as clinical or production thresholds.

A proper threshold calibration workflow would require a separate calibration split and an untouched evaluation split.

## Reproducibility note

Raw CheXpert images, local paths, sample-level labels, prediction files, threshold sweep CSV files, and Kaggle private outputs were intentionally not committed to the repository.

Only this sanitized summary report and the reusable threshold sweep script are stored in GitHub.
