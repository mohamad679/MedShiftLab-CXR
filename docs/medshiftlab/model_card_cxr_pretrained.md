# Model Card: Pretrained Chest X-ray Baseline

## Role in MedShiftLab-CXR

This model card documents the pretrained chest X-ray baseline used in MedShiftLab-CXR.

The model is used as a research instrument for studying how annotation uncertainty, dataset curation, calibration, robustness, and external dataset shift affect model behavior.

The project does not claim that the model is clinically validated or suitable for diagnosis.

## Intended Use

The model is intended for:

- Research evaluation
- Data-centric robustness analysis
- Calibration analysis
- Internal versus external validation comparison
- Failure-mode analysis
- Reproducible medical computer vision experimentation

The model is not intended for clinical decision-making, triage, diagnosis, treatment planning, or autonomous medical reporting.

## Model Family

The first implementation prioritizes pretrained chest X-ray models.

Preferred starting point:

- TorchXRayVision pretrained chest X-ray model

Optional extension only if integration remains simple and scientifically useful:

- RAD-DINO frozen feature extraction with a simple linear head

The project does not train a foundation model from scratch.

## Task

Multi-label chest X-ray classification.

Core pathology labels:

- Atelectasis
- Cardiomegaly
- Pleural Effusion
- Pneumonia
- Pneumothorax

No Finding is analyzed separately and is not treated as an equivalent pathology label.

## Input

Expected input modality:

- Chest X-ray image

The implementation should document:

- image format
- preprocessing
- resizing
- normalization
- view-position filtering if applied
- missing or invalid image handling

## Output

Expected output:

- label-wise prediction scores or probabilities for the core pathology labels

Outputs are model scores for research evaluation only. They are not clinical findings.

## Dataset Usage

CheXpert is used for:

- Internal evaluation
- Uncertainty strategy comparison
- Threshold selection
- Optional calibration fitting

VinDr-CXR is used for:

- Strict external validation
- Calibration degradation analysis
- Cross-dataset robustness analysis
- Failure analysis

VinDr-CXR must not be used for training, threshold tuning, calibration fitting, model selection, or uncertainty strategy selection.

## Training and Fine-Tuning Policy

Allowed:

- Pretrained inference
- Frozen feature extraction
- Linear probing if feasible and documented
- Threshold selection using CheXpert validation only
- Calibration fitting using CheXpert validation only

Not allowed in the first implementation:

- Training a foundation model from scratch
- Inventing a new architecture
- Large-scale fine-tuning
- Tuning on VinDr-CXR
- Claiming state-of-the-art performance
- Claiming clinical diagnostic validity

## Evaluation Metrics

Primary metrics:

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

Accuracy is not the primary metric because multi-label chest X-ray datasets are imbalanced.

## Calibration

Calibration is a first-class evaluation target.

Calibration fitting, if performed, must use CheXpert validation data only.

Calibration must be reported separately for:

- CheXpert internal evaluation
- VinDr-CXR external validation

A model may have acceptable discrimination but poor calibration. This distinction must be reported clearly.

## Robustness and Shift Analysis

The model may be evaluated under simple stress tests:

- brightness shift
- contrast shift
- mild Gaussian noise
- resolution or downsampling shift

Cross-dataset shift is evaluated by comparing CheXpert internal performance with VinDr-CXR external performance.

## Failure Analysis

Failure analysis should include:

- false positives
- false negatives
- high-confidence errors
- miscalibrated predictions
- failures associated with uncertain labels
- failures associated with external dataset shift

Qualitative examples, if included, are illustrative only and not clinical interpretation.

## Known Limitations

- Pretrained model behavior depends on original training data and preprocessing.
- Label definitions may differ across datasets.
- Calibration can degrade under dataset shift.
- External validation on VinDr-CXR does not prove prospective clinical safety.
- The model is not a medical device.
- The model is not clinically validated.
- The project does not claim radiologist-level performance.
- The project does not claim generalization to all hospitals or imaging settings.

## Allowed Claims

MedShiftLab-CXR may claim that the pretrained model is used to study data-centric robustness, calibration, and failure modes under annotation uncertainty and dataset shift.

## Disallowed Claims

MedShiftLab-CXR must not claim that:

- The model diagnoses patients.
- The model is clinically validated.
- The model is safe for clinical deployment.
- The model is better than radiologists.
- The model is state of the art.
- The model generalizes to all hospitals.
- External validation is equivalent to prospective clinical validation.

## Status

Draft model card for the first MedShiftLab-CXR implementation.
