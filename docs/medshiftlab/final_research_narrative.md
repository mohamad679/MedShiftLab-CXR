# MedShiftLab-CXR: Data-Centric Evaluation of Pretrained Chest X-Ray Foundation Models Under Annotation Uncertainty and Distribution Shift

## Abstract

MedShiftLab-CXR addresses a central problem in medical computer vision: a pretrained model's apparent performance can depend strongly on uncertain labels, dataset construction, and the population or acquisition domain used for evaluation. The project implements a reproducible chest X-ray research scaffold that separates validated metadata and label handling, model-independent prediction contracts, label-wise performance and calibration metrics, and structured experiment reporting. Tracked aggregate artifacts document a prior standalone TorchXRayVision run over a 1,000-image frontal CheXpert subset. That pre-freeze smoke/subset record is not a completed benchmark, external validation, clinical validation, or integrated package-level inference pipeline.

## Motivation

Pretrained medical imaging models can fail even when their architecture is unchanged. Annotation uncertainty can alter targets and apparent error patterns; dataset curation choices can change prevalence, exclusions, and case mix; and cross-dataset shift can expose differences in populations, scanners, acquisition protocols, and labeling practice. Aggregate discrimination metrics alone can also hide poor probability calibration, subgroup failures, or sensitivity to small input changes. A credible evaluation therefore needs explicit data assumptions, uncertainty policies, internal-versus-external separation, calibration measures, and reproducible artifacts.

## Research Question

> “How do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray foundation models?”

## Proposed Study Design

- Use CheXpert as the internal dataset for metadata audit, uncertainty analysis, and internal evaluation.
- Compare four explicit CheXpert uncertainty strategies: U-ignore, U-zero, U-one, and U-soft.
- Use MIMIC-CXR-JPG and/or VinDr-CXR only as external validation candidates under dataset shift.
- Do not use an external candidate for tuning, calibration fitting, model selection, threshold selection, uncertainty-strategy selection, or post-result protocol editing.
- Integrate a pretrained CXR model through the model adapter contract before considering broader model comparisons.
- Report label-wise AUROC, AUPRC, Brier score, expected calibration error (ECE), F1, sensitivity, and specificity.
- Preserve dataset, split, model, uncertainty-strategy, threshold, and ECE-bin provenance in each evaluation report.

## Implemented Framework

### Data and Label Layer

The repository defines a conservative shared CXR label ontology, explicit CheXpert uncertainty transformations, validated CheXpert and VinDr-CXR metadata schemas, a CheXpert CSV metadata loader, and dataset summary utilities. These components operate on metadata and labels only; they do not load images.

### Evaluation Layer

The evaluation layer computes the planned discrimination, probability-error, calibration, and threshold metrics from supplied targets and scores. It handles missing and soft targets explicitly, provides a row-based table interface, and produces a structured `EvaluationReport` with run metadata and label-wise results. Thresholds are provided inputs rather than tuned values.

### Model Adapter Boundary

Validated `PredictionRecord` and `PredictionBatch` schemas define the model-independent output contract. `CXRModelAdapter` specifies adapter behavior, a deterministic mock adapter supports tests, and the optional TorchXRayVision boundary maps precomputed model-output columns into the common schema. The prediction-to-evaluation bridge joins records by `image_id` and rejects missing, duplicate, or unknown identities.

### Experiment and Reporting Layer

The in-memory runner connects supplied records, adapter predictions, evaluation rows, and reports. The file-exporting runner writes a complete JSON report and stable label-wise CSV output. Neither runner downloads data or weights, loads images, or trains models.

### Reproducibility and Test Boundary

The focused local test script verifies the MedShiftLab-CXR data, evaluation, adapter, bridge, reporting, and experiment layers without invoking the legacy NeuroSight application suite or requiring TorchXRayVision runtime execution:

```bash
bash scripts/run_medshiftlab_tests.sh
```

## Scientific Contribution

MedShiftLab-CXR is not a new model architecture and is not a clinical deployment system. Its contribution is a reproducible, data-centric evaluation scaffold in which label uncertainty, dataset provenance, model outputs, evaluation settings, and external-validation boundaries are explicit and auditable. The intended scientific emphasis is on how annotation policy and dataset shift affect discrimination, calibration, robustness, and failure behavior rather than on claiming architectural novelty or model superiority.

## Fit to a Medical Computer Vision PhD

The project is grounded in radiological imaging through a focused multi-label chest X-ray study. It supports foundation-model evaluation without assuming that pretraining guarantees reliable transfer. Its explicit label and metadata contracts make data quality a first-class experimental variable, while the locked CheXpert/VinDr-CXR separation creates a defensible design for studying distribution shift. Calibration metrics, planned robustness and failure analyses, leakage controls, structured reports, and a focused test boundary align the work with trustworthy AI and reproducible medical imaging research.

## Current Limitations

- Standalone subset inference exists, but reusable package-level image loading and integrated adapter inference are incomplete.
- No completed CheXpert benchmark or external validation is reported.
- No model training is implemented or claimed.
- No clinical validation has been performed.
- No medical-device, FDA, CE, or other regulatory claim is made.
- TorchXRayVision is currently an optional adapter boundary, not a completed inference pipeline.

## Next Steps

1. Connect authorized CheXpert metadata and images without committing private or licensed data.
2. Configure authorized paths only in the ignored local registry configuration.
3. Add reusable package-level image loading and preprocessing.
4. Integrate pretrained prediction through the adapter contract.
5. Compare U-ignore, U-zero, U-one, and U-soft on the locked CheXpert internal protocol.
6. Evaluate the frozen study configuration on MIMIC-CXR-JPG and/or VinDr-CXR as strict external validation.
7. If calibration fitting is used, fit it only on CheXpert validation data and keep external candidates untouched.
8. Generate reliability diagrams, confidence summaries, and label-wise failure analyses.
9. Consider CT or MRI extensions only after the CXR framework has been evaluated under its frozen CXR protocol.
