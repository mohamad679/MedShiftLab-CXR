# Dataset Card: CheXpert

## Role in MedShiftLab-CXR

CheXpert is the primary internal dataset for MedShiftLab-CXR.

It is used to study how annotation uncertainty and data-curation decisions influence the robustness, calibration, and failure modes of pretrained chest X-ray AI models.

## Dataset Role

CheXpert is used for:

- Metadata audit
- Label distribution analysis
- Annotation uncertainty analysis
- Internal evaluation
- Threshold selection
- Optional calibration fitting
- Failure analysis under uncertainty-label groups

CheXpert is not used to claim clinical diagnostic validity.

## Modality

Chest X-ray.

## Task Type

Multi-label chest X-ray classification.

A single radiograph may contain multiple findings. Therefore, the project does not treat the task as single-label classification.

## Core Labels Used in MedShiftLab-CXR

The first implementation uses a conservative subset of labels that can be mapped to VinDr-CXR for external validation:

- Atelectasis
- Cardiomegaly
- Pleural Effusion
- Pneumonia
- Pneumothorax

No Finding is analyzed separately and is not treated as an equivalent pathology label.

The canonical mapping is defined in:

configs/labels/cxr_common_labels.yaml

## Annotation Uncertainty

CheXpert is central to this project because it includes uncertain labels.

MedShiftLab-CXR evaluates four uncertainty strategies:

- U-ignore: uncertain labels are excluded from supervised metric computation for that label.
- U-zero: uncertain labels are treated as negative.
- U-one: uncertain labels are treated as positive.
- U-soft: uncertain labels are treated as soft labels, for example 0.5.

These strategies are modeling choices. They must not be interpreted as clinical truth.

## Planned Split Policy

Whenever patient identifiers are available, splits must be patient-disjoint.

The project must avoid patient-level leakage between:

- training
- validation
- internal test

Image-level separation alone is not sufficient if multiple images from the same patient can appear across splits.

## Use in Evaluation

CheXpert may be used for:

- Internal performance evaluation
- Uncertainty strategy comparison
- Threshold selection
- Optional calibration fitting
- Internal failure analysis

CheXpert must not be used to make claims about external generalization by itself.

## Calibration Policy

Calibration fitting, if performed, must use CheXpert validation data only.

Calibration results must be reported separately for:

- internal CheXpert evaluation
- external VinDr-CXR validation

## Data Access and Storage

Raw CheXpert data are not included in this repository.

Users must obtain the dataset according to the original dataset access conditions and license.

Local raw data, processed images, and derived sensitive outputs must not be committed to the public repository.

## Expected Local Paths

The project may support local paths such as:

data/raw/chexpert/
data/processed/chexpert/
artifacts/chexpert/

These paths are local-only and should remain ignored by Git unless they contain non-sensitive synthetic fixtures or small schema examples.

## Known Limitations

- Labels are derived from radiology-report processing and may contain noise.
- Uncertain labels do not have a single clinically definitive interpretation.
- Label prevalence may differ substantially from external datasets.
- Internal performance on CheXpert does not prove external clinical reliability.
- The project does not claim that CheXpert labels are perfect ground truth.
- The project does not claim clinical diagnostic validity.

## Allowed Claims

MedShiftLab-CXR may claim that CheXpert is used for internal uncertainty-label analysis and data-centric evaluation.

## Disallowed Claims

MedShiftLab-CXR must not claim that:

- The model is clinically validated on CheXpert.
- CheXpert internal performance proves real-world clinical utility.
- Uncertain labels have one universally correct transformation.
- Results generalize to all hospitals or imaging settings.
- The system is a diagnostic product or medical device.

## Status

Draft dataset card for the first MedShiftLab-CXR implementation.
