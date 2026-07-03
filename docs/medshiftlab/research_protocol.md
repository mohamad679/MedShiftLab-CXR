# MedShiftLab-CXR Research Protocol

## Project Title

MedShiftLab-CXR: A Data-Centric Research Framework for Robust and Trustworthy Chest X-ray AI Under Annotation Uncertainty and Dataset Shift

## 1. Research Motivation

Medical imaging AI systems are often evaluated mainly by predictive performance on curated internal test sets. However, clinical reliability depends not only on model architecture, but also on data quality, annotation uncertainty, dataset composition, calibration, and distribution shift.

MedShiftLab-CXR focuses on these data-centric factors in chest X-ray AI. The project does not aim to invent a new neural network architecture. Its contribution is a reproducible research framework for evaluating how dataset-level decisions influence robustness, calibration, and failure modes of pretrained chest X-ray models.

## 2. Main Research Question

How do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray foundation models?

## 3. Sub-Questions

### RQ1: Annotation Uncertainty

How do different CheXpert uncertainty-label handling strategies affect model performance, calibration, and failure behavior?

Planned strategies:

- U-ignore: exclude uncertain labels from supervised metric computation for that label.
- U-zero: treat uncertain labels as negative.
- U-one: treat uncertain labels as positive.
- U-soft: treat uncertain labels as soft labels, for example 0.5.

### RQ2: Dataset Distribution

How do CheXpert and VinDr-CXR differ in label prevalence, available metadata, and image/domain characteristics?

### RQ3: External Generalization

How much do performance and calibration degrade when a pretrained chest X-ray model is evaluated under cross-dataset shift?

Primary validation direction:

- CheXpert internal evaluation
- VinDr-CXR strict external validation

### RQ4: Failure Analysis

Which labels, uncertainty groups, and dataset conditions are associated with false positives, false negatives, high-confidence errors, and miscalibration?

## 4. Scope

The first implementation is intentionally limited to chest X-ray classification.

In scope:

- Chest X-ray only
- CheXpert as the primary internal dataset
- VinDr-CXR as the strict external validation dataset
- Conservative common-label ontology
- Annotation uncertainty analysis
- Dataset curation analysis
- Pretrained CXR model evaluation
- Calibration analysis
- Distribution-shift analysis
- Simple robustness stress tests
- Failure analysis
- Reproducible documentation, configs, scripts, and tests

Out of scope for the first implementation:

- New neural network architecture design
- Training a foundation model from scratch
- CT implementation
- MRI implementation
- EEG or cognitive-score pipelines
- Full-stack application development
- FastAPI or frontend expansion
- Federated learning
- LangGraph agent work
- Clinical diagnosis claims
- Medical-device or regulatory claims
- State-of-the-art performance claims

## 5. Datasets

### CheXpert

CheXpert is the primary internal dataset because it provides large-scale chest X-ray labels with explicit uncertainty annotations.

Planned roles:

- Metadata audit
- Label distribution analysis
- Uncertainty-label strategy comparison
- Internal evaluation
- Threshold selection
- Optional calibration fitting

Raw data are not included in this repository. Users must obtain datasets according to the original access rules and licenses.

### VinDr-CXR

VinDr-CXR is the strict external validation dataset.

Planned roles:

- External validation
- Cross-dataset label distribution comparison
- Calibration degradation analysis
- Failure analysis
- Optional qualitative review using available annotations

VinDr-CXR must not be used for training, threshold tuning, calibration fitting, model selection, or uncertainty strategy selection.

### Future Datasets

The following datasets are future extensions only:

- MIMIC-CXR
- CT-RATE
- BraTS
- FastMRI

They are not required for the first validated implementation.

## 6. Label Protocol

The first implementation uses a conservative shared label set across CheXpert and VinDr-CXR.

Core pathology labels:

- Atelectasis
- Cardiomegaly
- Pleural Effusion
- Pneumonia
- Pneumothorax

No Finding is analyzed separately and is not treated as an equivalent pathology label.

The label mapping is stored in configs/labels/cxr_common_labels.yaml.

All mappings must remain explicit, reviewable, and documented with limitations. The project does not claim that labels are perfectly equivalent across datasets.

## 7. Evaluation Protocol

The primary task is multi-label chest X-ray classification.

CheXpert is used for internal evaluation, uncertainty strategy comparison, threshold selection, and optional calibration fitting.

VinDr-CXR is reserved for strict external validation. It must not be used for training, fine-tuning, threshold tuning, calibration fitting, model selection, or uncertainty strategy selection.

Splits must be patient-disjoint whenever patient identifiers are available. Image-level leakage across train, validation, and test splits is not acceptable.

## 8. Model Protocol

The first implementation uses pretrained chest X-ray models.

Preferred starting point:

- TorchXRayVision pretrained CXR model

Optional extension only if integration remains simple:

- RAD-DINO frozen feature extraction with a simple linear head

Allowed operations:

- Pretrained inference
- Frozen feature extraction
- Linear probing, if feasible
- Threshold selection on CheXpert validation only
- Calibration fitting on CheXpert validation only

Not allowed:

- Architecture invention
- Large-scale foundation model training
- Claiming model superiority or SOTA
- Tuning on external validation data

## 9. Metrics

Primary metrics are reported per label:

- AUROC
- AUPRC
- Brier score
- Expected Calibration Error

Secondary metrics:

- F1
- Sensitivity
- Specificity
- Reliability diagrams
- Confidence histograms

Accuracy is not treated as the primary metric because multi-label chest X-ray datasets are imbalanced and accuracy can be misleading.

## 10. Calibration Analysis

Calibration is a first-class research output.

Planned outputs:

- Label-wise ECE
- Macro ECE
- Brier score
- Reliability diagrams
- Confidence histograms

Calibration fitting, if performed, must use CheXpert validation data only. VinDr-CXR must remain untouched until external evaluation.

## 11. Distribution-Shift Analysis

The project studies shift at three levels:

- Label distribution shift between CheXpert and VinDr-CXR
- Metadata shift, depending on available fields
- Image or embedding shift using simple statistics or pretrained-model embeddings

Unavailable metadata must be documented rather than guessed or imputed without justification.

## 12. Robustness Stress Tests

Only simple, interpretable stress tests are included in the first implementation:

- Brightness shift
- Contrast shift
- Mild Gaussian noise
- Resolution or downsampling shift

Out of scope:

- Adversarial attacks
- GAN-based augmentation
- Diffusion-based augmentation
- Synthetic pathology generation
- Large augmentation search

## 13. Failure Analysis

Failure analysis focuses on:

- False positives
- False negatives
- High-confidence errors
- Miscalibrated predictions
- Errors associated with uncertain labels
- Errors associated with external dataset shift

Qualitative image examples, if included, are illustrative only and must not be presented as clinical interpretation.

## 14. Claims Allowed

The project may claim:

- It is a reproducible data-centric evaluation framework.
- It studies pretrained chest X-ray AI under annotation uncertainty and dataset shift.
- It uses CheXpert for internal uncertainty analysis.
- It uses VinDr-CXR for strict external validation.
- It prioritizes transparency, reproducibility, and honest limitations.

## 15. Claims Not Allowed

The project must not claim:

- Clinical validation
- Diagnostic performance for patient care
- Medical-device readiness
- Radiologist-level performance
- State-of-the-art performance
- Perfect label equivalence across datasets
- Generalization to all hospitals
- Full CT or MRI support in the first implementation

## 16. Success Criterion

The first implementation is successful if it produces a clean, reproducible, scientifically honest framework showing how annotation uncertainty and dataset shift affect pretrained chest X-ray model reliability.

It is unsuccessful if it becomes a broad demo with many features but no focused scientific argument.
