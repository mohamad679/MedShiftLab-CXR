# CheXpert/VinDr Reference Comparison Closeout v0.7.2

## Status

v0.7.2 closes the CheXpert/VinDr reference-comparison phase.

This stage does not train a model and does not run new inference. It summarizes the completed CheXpert reference evaluation and CheXpert-versus-VinDr cross-dataset comparison.

## Completed Chain

| Version | Purpose | Status |
|---|---|---|
| v0.7.0 | CheXpert frontal validation reference evaluation | complete |
| v0.7.1 | CheXpert vs VinDr cross-dataset metric comparison | complete |
| v0.7.2 | Reference-comparison closeout | complete |

## Core Result

The same TorchXRayVision DenseNet121 model was evaluated on:

| Dataset | Images | Role |
|---|---:|---|
| CheXpert frontal validation | 202 | reference/internal-style comparison point |
| VinDr/VinBigData | 15,000 | external-validation dataset |

## Main Cross-Dataset Findings

Cardiomegaly and Pleural Effusion are the strongest and most stable labels across CheXpert and VinDr/VinBigData.

Atelectasis has similar AUROC across datasets but much weaker AUPRC and F1 on VinDr/VinBigData, indicating prevalence-sensitive performance degradation.

Pneumothorax remains difficult to interpret because positive support is low and AUPRC/F1 remain weak despite moderate ranking performance in some settings.

Default threshold 0.5 is not a reliable universal operating point. Threshold analysis and calibration analysis are necessary for responsible reporting.

## Project Meaning

The project now has:

- real CheXpert reference evaluation
- real VinDr/VinBigData external inference
- real VinDr/VinBigData external metrics
- threshold analysis
- calibration analysis
- cross-dataset comparison

This is enough to support a credible cross-dataset shift narrative without claiming clinical validation.

## Boundary

This phase does not establish:

- clinical validation
- deployment readiness
- fairness validation
- prospective validation
- SOTA performance
- full benchmark completion

## Repository Boundary

No raw images, private manifests, private predictions, private metric CSV files, local runtime files, model weights, or Kaggle outputs are committed.

Only sanitized reports and README updates are tracked.

## Phase Decision

The CheXpert/VinDr reference-comparison phase is complete.

The next project direction should be one of:

1. final reproducibility packaging
2. model card and dataset card update
3. subgroup/fairness audit
4. MIMIC-CXR external candidate preparation
5. paper-style results section
