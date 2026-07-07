# MedShiftLab-CXR Manuscript Outline

## Working title

MedShiftLab-CXR: A Reproducible Framework for Studying Annotation Uncertainty and Dataset Shift in Pretrained Chest X-Ray Models

## Abstract skeleton

### Background

Pretrained chest X-ray models are often discussed through aggregate performance, but reliability under annotation uncertainty and cross-dataset shift remains difficult to characterize reproducibly.

### Objective

Evaluate how explicit uncertainty policies, dataset curation choices, and external cohort differences affect standardized chest X-ray prediction analysis under a conservative protocol.

### Methods

Describe the repository as a reproducible infrastructure package with prespecified ontology, uncertainty strategies, dataset roles, prediction schema, evaluation pipeline, and bounded local/manual preparation utilities for internal and external cohorts.

### Results

Do not populate with fabricated values. Insert only reviewed results from future authorized local runs that follow the frozen protocol.

### Conclusions

Keep claims limited to what the reviewed results support. Do not claim clinical readiness, deployment safety, or state-of-the-art performance without direct evidence.

## Introduction

- Motivation for data-centric evaluation in medical imaging
- Risks from label uncertainty and dataset shift
- Need for conservative reproducibility and leakage control
- Positioning of MedShiftLab-CXR as framework plus protocol, not a product

## Methods

### 1. Study design and protocol freeze

- Research question
- Predefined allowed/disallowed claims
- Internal versus external dataset roles

### 2. Datasets

- CheXpert as development/internal protocol dataset
- MIMIC-CXR-JPG and VinDr-CXR as external-validation candidates
- Access, licensing, and local/private handling assumptions

### 3. Label ontology and harmonization

- Conservative shared label set
- Handling of `No Finding`
- Dataset-specific harmonization references

### 4. Uncertainty handling

- `U-ignore`
- `U-zero`
- `U-one`
- `U-soft`
- Boundary that these are evaluation/modeling policies, not clinical truth

### 5. Prediction infrastructure

- Standardized prediction schema
- Adapter interface
- TorchXRayVision manual baseline boundary
- Safe adapter registry behavior

### 6. Evaluation infrastructure

- AUROC, AUPRC, F1, sensitivity, specificity, Brier score, ECE
- Strict label CSV requirements
- JSON/CSV reporting bundle

### 7. Internal protocol preparation

- Patient-level split assignment
- Uncertainty-strategy label-table generation
- Versioned config references

### 8. External-validation setup preparation

- Manifest validation
- Label harmonization
- Optional patient-overlap checks
- No threshold tuning or protocol edits after results

### 9. Robustness and calibration analysis plan

- Calibration-bin summaries
- Subgroup analysis
- Deterministic percentile bootstrap intervals for supported scalar metrics
- Non-clinical failure/degradation flags

## Evaluation plan

- Freeze dataset, label mapping, preprocessing, thresholds, and artifact schema before confirmatory runs
- Run internal evaluation on patient-disjoint CheXpert splits only after preparation is frozen
- Run external evaluation on prepared MIMIC-CXR-JPG and/or VinDr-CXR cohorts without retuning on external outcomes
- Report undefined metrics as undefined rather than imputing
- Keep exploratory analyses labeled as exploratory

## Results section placeholder

- Cohort accounting
- Label-wise internal evaluation tables
- External evaluation tables
- Calibration summaries
- Subgroup summaries
- Failure/degradation summaries

Only populate this section from actual reviewed artifacts.

## Limitations

- Infrastructure package does not equal completed benchmark
- Dataset label equivalence is imperfect
- External results are not interchangeable across cohorts
- No clinical deployment or prospective validation
- Manual/local workflows depend on authorized private data access

## Discussion prompts

- What changed when uncertainty handling changed?
- Which findings were stable across datasets and which were not?
- Where did calibration or coverage degrade?
- Which limitations remain unresolved because the package intentionally defers training and productization?

## Reproducibility and artifact checklist

- Frozen protocol version and Git commit
- Dataset cohort definitions
- Label harmonization config
- Preprocessing specification
- Prediction schema version
- Evaluation settings
- Threshold policy
- Bootstrap settings
- Output locations and checksums where appropriate
- Statement that private data, weights, and restricted outputs remain outside Git

## Claim boundary reminder

Do not fabricate numbers, tables, effect sizes, or conclusions. This outline is a packaging scaffold for a future manuscript, not evidence of completed results.
