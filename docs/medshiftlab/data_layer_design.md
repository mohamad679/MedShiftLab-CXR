# MedShiftLab-CXR Data Layer Design

## Purpose

The MedShiftLab-CXR data layer defines how public chest X-ray datasets are represented, validated, mapped, and prepared for data-centric evaluation.

The data layer is intentionally focused. It does not attempt to build a general medical imaging platform.

## Design Goals

The data layer must support:

- explicit dataset identity
- explicit label mapping
- uncertainty-label handling
- patient-disjoint split validation
- metadata audit
- reproducible dataset summaries
- internal versus external dataset separation
- safe local-only raw data handling

## Non-Goals

The first implementation does not support:

- clinical data ingestion from hospitals
- DICOMweb integration
- FHIR export
- frontend upload workflows
- real-time serving
- CT or MRI datasets
- EEG or cognitive-score datasets
- automatic private-data discovery
- committing raw medical images to Git

## Core Dataset Roles

### CheXpert

Role: primary internal dataset.

Allowed uses:

- metadata audit
- label distribution analysis
- uncertainty-label strategy comparison
- internal evaluation
- threshold selection
- optional calibration fitting

### VinDr-CXR

Role: strict external validation dataset.

Allowed uses:

- external validation
- label distribution shift analysis
- calibration degradation analysis
- failure analysis

VinDr-CXR must not be used for training, threshold tuning, calibration fitting, model selection, or uncertainty-strategy selection.

## Core Concepts

### Dataset Record

A dataset record represents one image-level sample and its metadata.

Required conceptual fields:

- dataset_name
- image_id
- image_path
- patient_id, if available
- split, if defined
- view_position, if available
- labels
- uncertainty information, if available

### Label Ontology

The label ontology defines the conservative shared labels used by the project.

The source of truth is:

configs/labels/cxr_common_labels.yaml

The ontology must make mappings explicit and reviewable.

### Uncertainty Strategy

CheXpert uncertainty values are handled by predefined strategies:

- U-ignore
- U-zero
- U-one
- U-soft

These are experimental modeling strategies, not clinical truth.

### Dataset Summary

Each dataset loader should be able to produce a summary including:

- number of records
- number of patients, if available
- label prevalence
- missing-label counts
- uncertain-label counts, if available
- view-position distribution, if available
- split distribution, if available

## Safety and Repository Hygiene

Raw medical images and real dataset exports must not be committed to the repository.

Local-only paths should remain under ignored directories such as:

- data/raw/
- data/processed/
- artifacts/

Only small synthetic fixtures, schema examples, or non-sensitive metadata examples may be committed.

## Implementation Boundaries

The first data-layer implementation should prioritize:

1. label ontology loading and validation
2. uncertainty strategy transformation
3. CheXpert metadata schema handling
4. VinDr-CXR metadata schema handling
5. dataset summary generation

It should not implement model inference, training, calibration, or visualization.

## Testing Requirements

The data layer must include tests for:

- label ontology validity
- required config fields
- uncertainty strategy behavior
- missing required metadata columns
- patient leakage detection when patient IDs are available
- external validation policy enforcement

## Success Criterion

The data layer is successful if it makes dataset identity, label mapping, uncertainty handling, and internal/external evaluation boundaries explicit, testable, and reproducible.
