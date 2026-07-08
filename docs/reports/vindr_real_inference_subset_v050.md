# VinDr/VinBigData Real Inference Subset v0.5.0

## Status

v0.5.0 records the first real VinDr/VinBigData image inference run.

This stage runs real model inference on a small VinDr subset. It does not compute external-validation metrics.

## Objective

The goal of v0.5.0 was to verify that the repository inference path can run a pretrained TorchXRayVision DenseNet model on real VinDr/VinBigData image files and produce usable prediction outputs.

## Inference Result

| Check | Result |
|---|---:|
| Model family | TorchXRayVision DenseNet121 |
| Weights | densenet121-res224-all |
| Device | cuda |
| Real VinDr images inferred | 25 |
| Raw model output shape | 25 × 18 |
| Project prediction rows | 25 |
| Project target labels | 5 |
| NaN values | 0 |
| Minimum mapped prediction | 0.00033947764 |
| Maximum mapped prediction | 0.8192774 |

## Project Label Mapping

The raw TorchXRayVision outputs were mapped to the project labels as follows:

| Project label | TorchXRayVision output |
|---|---|
| Atelectasis | pred_Atelectasis |
| Cardiomegaly | pred_Cardiomegaly |
| Pleural Effusion | pred_Effusion |
| Pneumonia | pred_Pneumonia |
| Pneumothorax | pred_Pneumothorax |

## Schema Validation

The mapped project prediction file passed prediction-schema validation in subset mode.

| Check | Result |
|---|---:|
| Prediction rows | 25 |
| Manifest rows available | 15,000 |
| Label rows available | 15,000 |
| Manifest-label overlap | 15,000 |
| Exact sample coverage required | no |
| Metrics generated | no |

## Boundary

This stage confirms real inference execution and prediction-schema compatibility.

It does not compute AUROC, AUPRC, F1, sensitivity, specificity, calibration, subgroup metrics, or clinical validation.

## Repository Boundary

No raw images, local manifests, local labels, local prediction files, metrics files, Kaggle paths, model weights, or private runtime outputs are committed.

Only this sanitized report is tracked.

## Next Stage

v0.5.1 should run real inference on the full VinDr/VinBigData prepared image set or a larger controlled batch.
