# CheXpert Reference Evaluation v0.7.0

## Status

v0.7.0 records a real CheXpert frontal validation reference evaluation for MedShiftLab-CXR.

This stage ran real TorchXRayVision inference and evaluation on CheXpert validation images. It does not train a model and does not establish clinical validation.

## Objective

The goal of v0.7.0 was to create an internal CheXpert reference point that can later be compared against the VinDr/VinBigData external-validation results.

## Evaluation Setup

| Field | Value |
|---|---:|
| Dataset | CheXpert validation |
| View subset | Frontal only |
| Validation rows | 234 |
| Frontal rows evaluated | 202 |
| Missing images | 0 |
| Model | TorchXRayVision DenseNet121 |
| Weights | densenet121-res224-all |
| Prediction rows | 202 |
| Metrics rows | 5 |
| Threshold | 0.5 |
| Calibration bins | 10 |

## Label Support

| Label | Positive binary labels |
|---|---:|
| Atelectasis | 75 |
| Cardiomegaly | 66 |
| Pleural Effusion | 64 |
| Pneumonia | 8 |
| Pneumothorax | 7 |

## Metrics

| Label | AUROC | AUPRC | F1 | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.808399 | 0.640529 | 0.583658 | 1.000000 | 0.157480 | 0.330932 | 0.381139 |
| Cardiomegaly | 0.808712 | 0.647801 | 0.556522 | 0.969697 | 0.264706 | 0.270682 | 0.305916 |
| Pleural Effusion | 0.881907 | 0.772327 | 0.538462 | 0.984375 | 0.224638 | 0.324979 | 0.400807 |
| Pneumonia | 0.825387 | 0.169421 | 0.088889 | 1.000000 | 0.154639 | 0.341256 | 0.535985 |
| Pneumothorax | 0.639560 | 0.057658 | 0.068293 | 1.000000 | 0.020513 | 0.295667 | 0.508433 |

## Interpretation

The CheXpert reference evaluation shows strong ranking performance for Pleural Effusion and moderate-to-good AUROC for Atelectasis, Cardiomegaly, and Pneumonia.

At the default threshold of 0.5, the model has very high sensitivity but low specificity across labels. This indicates substantial overprediction at the default operating point.

Pneumonia and Pneumothorax have low positive support in this frontal validation subset and should be interpreted cautiously.

## Boundary

This is a retrospective reference evaluation on CheXpert validation images.

It does not establish clinical validation, deployment readiness, fairness validation, prospective validation, or full benchmark completion.

## Repository Boundary

No raw images, local manifests, local predictions, local evaluation JSON, local evaluation CSV, model weights, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.7.1 should compare CheXpert reference metrics against VinDr/VinBigData external metrics to summarize cross-dataset performance shift.
