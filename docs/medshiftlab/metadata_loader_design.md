# MedShiftLab-CXR Metadata Loader Design

## Purpose

The metadata loader converts dataset metadata files into validated MedShiftLab-CXR records.

It is responsible for dataset structure, labels, uncertainty handling, and summary statistics.

It is not responsible for model inference, image loading, training, calibration, or visualization.

## First Implementation Scope

The first implementation supports:

- CheXpert metadata CSV loading
- CheXpert row-to-record parsing
- CheXpert uncertainty strategy application
- Dataset-level summary generation
- Basic schema validation
- Patient ID extraction
- Missing-column handling

VinDr-CXR metadata support will be added after CheXpert loading is stable.

## CheXpert Loader Requirements

The CheXpert loader must:

- accept a metadata CSV path
- validate that the file exists
- read rows using pandas
- parse each row into a CheXpertRecord
- apply a selected uncertainty strategy
- preserve raw labels
- produce project-level labels
- support small smoke fixtures for tests

## Dataset Summary Requirements

The summary generator should report:

- dataset name
- number of records
- number of patients if available
- number of records without patient ID
- label prevalence after uncertainty strategy
- missing label counts
- positive label counts
- negative label counts
- uncertain or ignored label counts

## Boundary Rules

The loader must not:

- load raw images
- preprocess pixels
- call pretrained models
- tune thresholds
- fit calibration
- use external validation data for model selection
- commit real dataset metadata or images to Git

## Testing Requirements

Tests must cover:

- successful loading of a tiny synthetic CheXpert metadata CSV
- missing CSV file handling
- missing required Path column handling
- uncertainty strategy behavior through the loader
- dataset summary correctness on a small fixture

## Success Criterion

The metadata loader is successful if a tiny CheXpert-style CSV can be loaded into validated records and summarized reproducibly without accessing raw medical images.
