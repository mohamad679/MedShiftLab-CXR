# MedShiftLab-CXR Research Narrative

## Research problem

A pretrained chest X-ray classifier can appear reliable on one dataset and degrade when label definitions, prevalence, acquisition conditions, or patient populations change. MedShiftLab-CXR makes those data and evaluation choices explicit so that annotation uncertainty and cross-dataset shift can be studied without conflating them with architecture development.

## Implemented project

The repository provides:

- a conservative chest X-ray label ontology and CheXpert/VinDr mappings
- explicit `U-ignore`, `U-zero`, `U-one`, and `U-soft` target materialization
- private/local data configuration and path-safety controls
- image loading and TorchXRayVision DenseNet121 inference workflows
- standardized prediction and evaluation schemas
- AUROC, AUPRC, Brier score, ECE, threshold metrics, calibration bins, bootstrap summaries, and subgroup audits
- versioned sanitized reports with private runtime artifacts kept outside Git

## Completed real-data evaluation chain

The `v1.0.0` release records two reference domains using the same pretrained DenseNet121 model:

1. A real CheXpert frontal validation reference evaluation on 202 images.
2. Full real inference on 15,000 prepared VinDr/VinBigData images, followed by external-dataset metrics for Atelectasis, Cardiomegaly, Pleural Effusion, and Pneumothorax.

The VinDr analysis also includes threshold selection, calibration-oriented analysis, operating-point recommendations, and a CheXpert/VinDr comparison. Pneumonia is excluded from the VinDr metrics because the conservative mapping produced no positive Pneumonia examples.

## Main interpretation

Cardiomegaly and Pleural Effusion were the most stable labels across the two evaluated datasets. Atelectasis and Pneumothorax retained useful ranking performance but showed weak precision-sensitive and calibration behavior under the large prevalence shift. The default threshold of 0.5 was not a universally suitable operating point.

These findings support a dataset-shift evaluation narrative. They do not establish clinical validity or identify a single causal mechanism for the observed performance differences.

## Annotation-uncertainty boundary

The data layer can generate all four uncertainty strategies, but the current release does not train four separate model heads or establish how uncertainty handling changes external generalization. A valid post-v1.0 experiment should keep the encoder fixed, fit lightweight heads under each strategy, and compare them on a frozen external protocol.

## Compute-aware design

The project deliberately avoids training a foundation model from scratch. The preferred next-stage design is to extract frozen pretrained embeddings once, store them privately, and run linear probes, calibration analysis, bootstrap intervals, and subgroup analyses on CPU.

## Claim boundaries

MedShiftLab-CXR does not claim:

- clinical or prospective validation
- diagnostic utility
- fairness validation
- deployment or regulatory readiness
- state-of-the-art or radiologist-level performance
- universal generalization
- completion of a comprehensive multi-model clinical benchmark

## Current release authority

The authoritative closeout is [`docs/reports/final_release_closeout_v100.md`](../reports/final_release_closeout_v100.md). Earlier phase and scaffold documents remain historical records of the repository state at the time they were written.
