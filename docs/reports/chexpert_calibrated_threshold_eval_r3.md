# CheXpert Safe-Subset R3 Calibrated Threshold Evaluation Report

## Status

This report records an exploratory calibration/evaluation split threshold experiment on the 202 frontal CheXpert validation images available in the local/private Kaggle run.

This is not a full benchmark, not external validation, and not clinical validation.

## Source run

- Runtime: Kaggle Notebook
- Dataset: CheXpert validation split
- Image subset: 202 frontal validation images
- Model: TorchXRayVision DenseNet121
- Preprocessing version: `phase5-baseline-inference-v2`
- Normalization: `torchxrayvision`
- Split method: deterministic greedy multilabel-balanced split
- Calibration records: 101
- Evaluation records: 101
- Threshold grid: 0.00 to 1.00 with step 0.01

## Split support

| Label | Total Positive | Calibration Positive | Evaluation Positive | Calibration Negative | Evaluation Negative |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 75 | 38 | 37 | 63 | 64 |
| Cardiomegaly | 66 | 33 | 33 | 68 | 68 |
| Pleural Effusion | 64 | 32 | 32 | 69 | 69 |
| Pneumonia | 8 | 4 | 4 | 97 | 97 |
| Pneumothorax | 7 | 4 | 3 | 97 | 98 |

## Calibration-selected thresholds

| Label | Threshold | Calibration F1 | Precision | Sensitivity | Specificity | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.68 | 0.833333 | 0.760870 | 0.921053 | 0.825397 | 0.873225 |
| Cardiomegaly | 0.65 | 0.750000 | 0.692308 | 0.818182 | 0.823529 | 0.820856 |
| Pleural Effusion | 0.70 | 0.806452 | 0.833333 | 0.781250 | 0.927536 | 0.854393 |
| Pneumonia | 0.68 | 0.153846 | 0.111111 | 0.250000 | 0.917526 | 0.583763 |
| Pneumothorax | 0.66 | 0.400000 | 1.000000 | 0.250000 | 1.000000 | 0.625000 |

## Evaluation aggregate comparison

| Threshold Source | Mean F1 | Mean Sensitivity | Mean Specificity | Mean Balanced Accuracy |
|---|---:|---:|---:|---:|
| Default 0.5 | 0.3287709977784775 | 1.0 | 0.0 | 0.5 |
| Calibration-selected | 0.344235351836277 | 0.39243447993448 | 0.7873732236058923 | 0.5899038517701862 |

## Evaluation results by label

| Label | Threshold Source | Threshold | F1 | Precision | Sensitivity | Specificity | Balanced Accuracy | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | Calibration-selected | 0.68 | 0.597701 | 0.520000 | 0.702703 | 0.625000 | 0.663851 | 26 | 24 | 40 | 11 |
| Atelectasis | Default 0.5 | 0.50 | 0.536232 | 0.366337 | 1.000000 | 0.000000 | 0.500000 | 37 | 64 | 0 | 0 |
| Cardiomegaly | Calibration-selected | 0.65 | 0.560976 | 0.469388 | 0.696970 | 0.617647 | 0.657308 | 23 | 26 | 42 | 10 |
| Cardiomegaly | Default 0.5 | 0.50 | 0.492537 | 0.326733 | 1.000000 | 0.000000 | 0.500000 | 33 | 68 | 0 | 0 |
| Pleural Effusion | Calibration-selected | 0.70 | 0.562500 | 0.562500 | 0.562500 | 0.797101 | 0.679801 | 18 | 14 | 55 | 14 |
| Pleural Effusion | Default 0.5 | 0.50 | 0.481203 | 0.316832 | 1.000000 | 0.000000 | 0.500000 | 32 | 69 | 0 | 0 |
| Pneumonia | Calibration-selected | 0.68 | 0.000000 | 0.000000 | 0.000000 | 0.917526 | 0.458763 | 0 | 8 | 89 | 4 |
| Pneumonia | Default 0.5 | 0.50 | 0.076190 | 0.039604 | 1.000000 | 0.000000 | 0.500000 | 4 | 97 | 0 | 0 |
| Pneumothorax | Calibration-selected | 0.66 | 0.000000 | 0.000000 | 0.000000 | 0.979592 | 0.489796 | 0 | 2 | 96 | 3 |
| Pneumothorax | Default 0.5 | 0.50 | 0.057692 | 0.029703 | 1.000000 | 0.000000 | 0.500000 | 3 | 98 | 0 | 0 |

## Interpretation

The calibrated thresholds improved specificity and mean balanced accuracy compared with the default 0.5 threshold. They also slightly improved mean F1 on the held-out evaluation split.

However, calibration-selected thresholds reduced sensitivity substantially, especially for rare labels. Pneumonia and Pneumothorax had very small positive support in both calibration and evaluation splits, so their threshold behavior is unstable and should not be interpreted as validated performance.

The result supports the need for a proper calibration protocol, but it remains exploratory and should not be used as a clinical or production operating point.

## Reproducibility note

Raw CheXpert images, local paths, sample-level labels, predictions, calibration split files, and Kaggle private outputs were intentionally not committed to the repository.

Only this sanitized summary report and the reusable calibrated-threshold evaluation script are stored in GitHub.
