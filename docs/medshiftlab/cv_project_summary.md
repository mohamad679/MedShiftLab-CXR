# MedShiftLab-CXR CV Project Summary

## One-line CV bullet

- Built MedShiftLab-CXR, a reproducible chest X-ray research scaffold for local/manual evaluation of pretrained models under annotation uncertainty and cross-dataset shift, with standardized prediction, evaluation, and reporting infrastructure.

## Short technical bullets

- Designed a conservative data and evaluation stack covering dataset registry controls, image-loading/preprocessing utilities, standardized prediction schemas, and strict JSON/CSV reporting.
- Implemented manual-only protocol preparation for CheXpert internal splits plus MIMIC-CXR-JPG and VinDr-CXR external-validation setup, with explicit leakage and claim boundaries.
- Added robustness, calibration-bin, subgroup, and bootstrap analysis scaffolding over existing prediction artifacts without overstating unrun experiments.

## Portfolio paragraph

MedShiftLab-CXR is a research-oriented software package for studying reliability questions in pretrained chest X-ray classifiers without collapsing infrastructure work into unsupported performance claims. The project packages explicit label harmonization, uncertainty-policy handling, private/local dataset configuration, bounded image loading, a standardized prediction contract, safe model-adapter boundaries, strict evaluation/report schemas, and downstream robustness-analysis scaffolding. The emphasis is on reproducibility, conservative documentation, and separation between implemented infrastructure and deferred confirmatory experiments.

## Skills and keywords

- Python
- Pydantic
- pandas
- scikit-learn
- reproducible research infrastructure
- medical imaging
- chest X-ray
- evaluation pipelines
- uncertainty handling
- calibration analysis
- subgroup analysis
- experiment packaging
- privacy-conscious data workflows

## Wording boundary

Use this summary as infrastructure and research-scaffold work. Do not rewrite it into claims about benchmark completion, external validation completion, clinical utility, or achieved model performance unless those claims are separately supported by reviewed artifacts.
