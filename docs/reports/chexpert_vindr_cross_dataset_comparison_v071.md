# CheXpert vs VinDr Cross-Dataset Comparison v0.7.1

## Status

v0.7.1 compares the CheXpert frontal validation reference evaluation against the VinDr/VinBigData external-validation evaluation.

This stage does not train a model and does not run new inference. It compares already documented metrics from v0.5.2 and v0.7.0.

## Objective

The goal of v0.7.1 is to summarize cross-dataset performance shift for the shared evaluated labels.

## Compared Runs

| Run | Dataset | Images | Model | Weights | Status |
|---|---|---:|---|---|---|
| v0.5.2 | VinDr/VinBigData | 15,000 | TorchXRayVision DenseNet121 | densenet121-res224-all | complete |
| v0.7.0 | CheXpert validation frontal | 202 | TorchXRayVision DenseNet121 | densenet121-res224-all | complete |

## Shared Label Comparison

| Label | CheXpert AUROC | VinDr AUROC | AUROC delta | CheXpert AUPRC | VinDr AUPRC | AUPRC delta | CheXpert F1 | VinDr F1 | F1 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.808399 | 0.798618 | -0.009781 | 0.640529 | 0.044507 | -0.596022 | 0.583658 | 0.086042 | -0.497616 |
| Cardiomegaly | 0.808712 | 0.887571 | 0.078859 | 0.647801 | 0.605342 | -0.042459 | 0.556522 | 0.566805 | 0.010283 |
| Pleural Effusion | 0.881907 | 0.894115 | 0.012208 | 0.772327 | 0.606659 | -0.165668 | 0.538462 | 0.566575 | 0.028113 |
| Pneumothorax | 0.639560 | 0.823805 | 0.184245 | 0.057658 | 0.040432 | -0.017226 | 0.068293 | 0.046148 | -0.022145 |

Deltas are computed as VinDr minus CheXpert.

## Key Findings

Cardiomegaly and Pleural Effusion are the most stable labels across the two evaluations. Both maintain strong AUROC on VinDr/VinBigData and have the most interpretable external-validation behavior.

Atelectasis shows similar AUROC across datasets but a severe AUPRC and F1 drop on VinDr/VinBigData. This suggests that ranking performance alone is not enough to describe the shift for this label.

Pneumothorax shows higher AUROC on VinDr/VinBigData but weak AUPRC and F1 on both datasets. This label remains difficult to interpret due to low positive support and threshold sensitivity.

CheXpert threshold 0.5 behavior showed high sensitivity and low specificity, while VinDr threshold analysis showed that some labels benefit from per-label thresholding.

## Interpretation

The comparison supports a clear cross-dataset shift narrative:

- ranking metrics can remain moderate or strong across datasets
- precision-sensitive metrics can collapse under prevalence shift
- default threshold 0.5 is not a reliable universal operating point
- calibration and threshold reports are necessary for responsible interpretation

## Boundary

This is a retrospective cross-dataset comparison of documented evaluation runs.

It does not establish clinical validation, deployment readiness, fairness validation, prospective validation, or SOTA performance.

## Repository Boundary

No raw images, private manifests, private predictions, private metric CSV files, local runtime files, model weights, or Kaggle outputs are committed.

Only this sanitized comparison report is tracked.

## Next Stage

v0.7.2 should close the CheXpert/VinDr reference-comparison phase and define the next project direction.
