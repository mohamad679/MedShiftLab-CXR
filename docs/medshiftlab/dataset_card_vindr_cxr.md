# Dataset Card: VinDr-CXR

## Role in MedShiftLab-CXR

VinDr-CXR is the strict external validation dataset for MedShiftLab-CXR.

It is used to evaluate how pretrained chest X-ray AI models behave under cross-dataset distribution shift after internal analysis on CheXpert.

## Dataset Role

VinDr-CXR is used for:

- External validation
- Cross-dataset label distribution comparison
- Calibration degradation analysis
- Distribution-shift analysis
- Failure analysis
- Optional qualitative review using available annotations

VinDr-CXR is not used for training, threshold tuning, calibration fitting, model selection, or uncertainty-strategy selection.

## Modality

Chest X-ray.

## Task Type

Multi-label chest X-ray classification.

A single radiograph may contain multiple findings. The project therefore treats the task as multi-label classification.

## Core Labels Used in MedShiftLab-CXR

The first implementation uses a conservative subset of labels that can be mapped from CheXpert to VinDr-CXR:

- Atelectasis
- Cardiomegaly
- Pleural Effusion
- Pneumonia
- Pneumothorax

No Finding is analyzed separately and is not treated as an equivalent pathology label.

The canonical mapping is defined in:

configs/labels/cxr_common_labels.yaml

## External Validation Policy

VinDr-CXR must remain a genuine external test distribution.

The following actions are not allowed on VinDr-CXR in the first implementation:

- Training
- Fine-tuning
- Threshold tuning
- Calibration fitting
- Model selection
- Uncertainty strategy selection
- Hyperparameter optimization

All model choices, thresholds, and optional calibration parameters must be determined using CheXpert internal data only before evaluating on VinDr-CXR.

## Use in Evaluation

VinDr-CXR is used to assess:

- External AUROC and AUPRC
- External Brier score and ECE
- Internal-to-external performance drop
- Calibration degradation under dataset shift
- Label prevalence differences
- Failure modes under external distribution shift

## Annotation Considerations

VinDr-CXR annotations may differ from CheXpert annotations in label definitions, annotation process, source population, acquisition patterns, and radiologist workflow.

Therefore, MedShiftLab-CXR does not claim that labels are perfectly equivalent across CheXpert and VinDr-CXR.

All cross-dataset comparisons must be interpreted as approximate, dataset-level evaluations rather than exact clinical equivalence studies.

## Data Access and Storage

Raw VinDr-CXR data are not included in this repository.

Users must obtain the dataset according to the original dataset access conditions and license.

Local raw data, processed images, and derived sensitive outputs must not be committed to the public repository.

## Expected Local Paths

The project may support local paths such as:

data/raw/vindr_cxr/
data/processed/vindr_cxr/
artifacts/vindr_cxr/

These paths are local-only and should remain ignored by Git unless they contain non-sensitive synthetic fixtures or small schema examples.

## Known Limitations

- VinDr-CXR is an external dataset, but it is not a substitute for prospective clinical validation.
- Label definitions may not perfectly match CheXpert.
- Dataset shift may reflect many factors, including acquisition protocol, population, annotation process, and preprocessing.
- Poor external performance does not identify a single causal mechanism by itself.
- Good external performance does not prove clinical deployment readiness.
- The project does not claim clinical diagnostic validity.

## Allowed Claims

MedShiftLab-CXR may claim that VinDr-CXR is used as an independent external validation dataset for evaluating robustness, calibration, and failure modes under dataset shift.

## Disallowed Claims

MedShiftLab-CXR must not claim that:

- The model is clinically validated on VinDr-CXR.
- VinDr-CXR performance proves real-world clinical utility.
- CheXpert and VinDr-CXR labels are perfectly equivalent.
- The system generalizes to all hospitals or acquisition settings.
- The system is a diagnostic product or medical device.

## Status

Draft dataset card for the first MedShiftLab-CXR implementation.
