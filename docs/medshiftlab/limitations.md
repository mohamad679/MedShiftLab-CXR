# MedShiftLab-CXR Limitations

## Purpose

MedShiftLab-CXR is a research framework for studying data-centric reliability of pretrained chest X-ray AI models.

It is not a clinical product, not a diagnostic system, and not a medical device.

## Core Limitations

The first implementation is limited to chest X-ray classification.

It does not implement:

- CT analysis
- MRI analysis
- EEG analysis
- cognitive-score modeling
- multimodal fusion
- clinical deployment
- prospective validation
- regulatory validation

## Dataset Limitations

The first implementation focuses on:

- CheXpert for internal uncertainty-label analysis
- VinDr-CXR for strict external validation

The project does not claim that these datasets represent all hospitals, scanners, populations, acquisition protocols, or clinical workflows.

## Label Limitations

The common-label ontology is conservative but imperfect.

CheXpert and VinDr-CXR may differ in:

- label definitions
- annotation process
- radiologist workflow
- report-derived versus image-level annotation behavior
- population and acquisition characteristics

The project does not claim perfect label equivalence across datasets.

## Uncertainty Limitations

CheXpert uncertainty-label strategies are modeling choices.

U-ignore, U-zero, U-one, and U-soft are not clinical truth. They are experimental strategies for studying how annotation uncertainty affects model behavior.

## Model Limitations

The project uses pretrained chest X-ray models as research instruments.

It does not claim:

- new architecture novelty
- state-of-the-art performance
- radiologist-level performance
- clinical diagnostic validity
- safety for patient care

## Evaluation Limitations

Internal evaluation on CheXpert and external validation on VinDr-CXR do not replace prospective clinical validation.

Good performance does not prove deployment readiness.

Poor performance does not identify a single causal mechanism without further analysis.

## Calibration Limitations

Calibration can degrade under dataset shift.

Calibration results are research outputs and must not be interpreted as clinical confidence for patient care.

## Explainability and Failure Analysis Limitations

Failure examples and qualitative visualizations, if included, are illustrative only.

They are not clinical interpretations and must not be used for diagnosis.

## Allowed Interpretation

MedShiftLab-CXR may be interpreted as a reproducible research framework for studying how data-centric decisions affect pretrained chest X-ray model reliability.

## Disallowed Interpretation

MedShiftLab-CXR must not be interpreted as:

- a diagnostic AI system
- a medical device
- a clinical decision-support tool
- evidence of deployment readiness
- evidence of universal generalization
- proof that one uncertainty strategy is clinically correct
