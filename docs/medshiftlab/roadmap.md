# MedShiftLab-CXR Roadmap

> **Document status:** Historical pre-v1.0 roadmap. Completed real-data work and current boundaries are recorded in the [final release closeout](../reports/final_release_closeout_v100.md).

## Project Direction

MedShiftLab-CXR is designed as a focused research track inside the NeuroSight repository.

The first implementation prioritizes scientific credibility, reproducibility, and data-centric analysis over feature breadth.

## First Implementation: 16-Day PhD Application Version

Primary goal:

Build a reproducible chest-X-ray-only framework for evaluating pretrained CXR AI under annotation uncertainty and dataset shift.

Core deliverables:

- Research protocol
- CheXpert dataset card
- VinDr-CXR dataset card
- Pretrained CXR model card
- Conservative common-label mapping
- Dataset configuration files
- CheXpert metadata ingestion
- VinDr-CXR metadata ingestion
- Uncertainty-label strategy implementation
- Internal evaluation protocol
- Strict external validation protocol
- Calibration metrics
- Distribution-shift analysis
- Robustness stress tests
- Failure-analysis report
- Reproducibility documentation
- Tests and CI-compatible checks

## Stage A: Documentation and Protocol Foundation

Status: in progress.

Tasks:

- Define research question
- Define dataset protocol
- Define label mapping
- Define uncertainty strategies
- Define evaluation protocol
- Define claims and limitations

## Stage B: Data Layer

Planned tasks:

- Implement CheXpert metadata loader
- Implement VinDr-CXR metadata loader
- Implement common label ontology validation
- Implement uncertainty-label transforms
- Implement dataset summary reports
- Add tests for schema and leakage rules

## Stage C: Model Layer

Planned tasks:

- Implement pretrained TorchXRayVision adapter
- Standardize preprocessing
- Produce label-wise prediction outputs
- Document model assumptions
- Add import and inference smoke tests

Optional only if simple:

- RAD-DINO frozen feature extraction

## Stage D: Evaluation Layer

Planned tasks:

- Implement AUROC and AUPRC
- Implement Brier score
- Implement Expected Calibration Error
- Implement reliability diagram data
- Implement threshold handling
- Implement internal and external evaluation reports

## Stage E: Data-Centric Analysis

Planned tasks:

- Compare CheXpert uncertainty strategies
- Analyze label prevalence shift
- Analyze calibration degradation
- Analyze robustness under simple stress tests
- Analyze high-confidence failures

## Stage F: Final PhD Application Package

Planned outputs:

- Updated README positioning MedShiftLab-CXR
- Research protocol summary
- Reproducibility guide
- Limitations section
- Figures and tables if data are available
- Final project narrative for PhD interview discussion

## Future Extensions

Future work may include:

- MIMIC-CXR as an additional external dataset
- CT-RATE for CT distribution-shift analysis
- BraTS for MRI segmentation robustness
- FastMRI for MRI reconstruction reliability
- MedSAM-based segmentation analysis
- More advanced calibration methods
- Subgroup robustness analysis
- Reader-study-inspired annotation disagreement analysis

These are future research directions and are not required for the first implementation.

## Explicit Non-Goals

The first implementation will not include:

- new foundation model training
- new neural architecture design
- full-stack application development
- clinical deployment
- medical-device claims
- CT or MRI implementation
- EEG or cognitive-score pipelines
- LangGraph agent expansion
- federated learning expansion
