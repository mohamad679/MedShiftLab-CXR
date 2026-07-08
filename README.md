# MedShiftLab-CXR

MedShiftLab-CXR is a reproducible research scaffold for studying pretrained chest X-ray classification models under annotation uncertainty and cross-dataset shift. The repository packages conservative data, prediction, evaluation, and reporting infrastructure for local/manual research workflows. It does not present a completed benchmark, external validation study, or clinical system.

## Objective

The project objective is to support a controlled research workflow around one question:

> How do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray models?

## Current implemented scope

- Research protocol and documentation for conservative claim boundaries
- Chest X-ray label ontology and explicit uncertainty handling
- Local/private dataset registry with a tracked null-only config template
- Reusable image loading, preprocessing, path-containment, and bounded smoke-test utilities
- Standardized prediction schema and model-adapter contract
- Safe adapter registry with mock adapters and a manual optional TorchXRayVision boundary
- Manual-only bounded baseline inference entry point for small authorized local subsets
- Standardized prediction evaluation for local JSON/CSV prediction artifacts plus matching label CSVs
- CheXpert internal protocol preparation with patient-level split manifests and uncertainty-strategy label tables
- External-validation setup preparation for `mimic_cxr_jpg` and `vindr_cxr`
- Robustness, calibration-bin, subgroup, bootstrap, and failure-flag scaffolding over existing prediction artifacts
- JSON/CSV reporting exports, in-memory runners, and focused tests

## Explicit limitations

- No full-dataset inference workflow is run by default
- No completed CheXpert benchmark
- No completed MIMIC-CXR-JPG or VinDr-CXR external validation
- No real benchmark results are committed as part of the package
- No training, fine-tuning, LoRA, or PEFT workflow
- No automatic model-weight download path
- No calibration fitting or calibration-plot rendering
- No hosted app, API, or clinical deployment surface

## What is not claimed

This repository does not claim:

- benchmark completion
- external validation completion
- clinical validation
- diagnostic utility
- deployment readiness
- state-of-the-art performance
- final model performance
- regulatory readiness

Tracked infrastructure or bounded local/manual scripts should not be interpreted as completed experiments.

## Safe setup

Create a Python 3.11 environment and install the package in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Only if you intentionally plan to exercise the manual optional TorchXRayVision baseline path:

```bash
pip install -e .[dev,torchxrayvision]
```

The repository does not include datasets, model weights, or private path configuration.

## Test command

```bash
bash scripts/run_medshiftlab_tests.sh
```

## Local/private data policy

- Keep raw medical data, restricted metadata, model weights, and local outputs outside Git-tracked artifacts.
- Copy `configs/data/example_local_paths.yaml` to `configs/data/local_paths.yaml` locally and populate only the ignored copy.
- Use relative image paths inside manifests and label tables where the repository expects them.
- Do not commit private absolute paths, prediction artifacts, evaluation artifacts, or credentials.

## Manual workflow overview

1. Freeze the intended study boundary in [the research protocol](docs/medshiftlab/research_protocol.md).
2. Populate the ignored local config from [configs/data/example_local_paths.yaml](configs/data/example_local_paths.yaml).
3. Validate local dataset-path configuration and run a bounded image-loading smoke test.
4. If authorized, run bounded manual baseline inference on a small manifest.
5. Evaluate standardized prediction files against matching local label tables.
6. Prepare internal or external protocol artifacts before any real larger run.
7. Run robustness/calibration/subgroup analysis only on existing local prediction artifacts.

Command examples are in [docs/medshiftlab/reproducibility_guide.md](docs/medshiftlab/reproducibility_guide.md).

## Reproducibility commands

Minimal verification:

```bash
bash scripts/run_medshiftlab_tests.sh
```

Manual/local workflow commands for registry validation, image smoke tests, bounded inference, evaluation, protocol preparation, and robustness analysis are documented in [docs/medshiftlab/reproducibility_guide.md](docs/medshiftlab/reproducibility_guide.md).

## Repository structure

```text
src/medshiftlab/                     Core data, model-boundary, evaluation, experiment, and reporting modules
scripts/                             Manual/local CLI entry points
configs/                             Path-free tracked configs, protocol YAML, label mappings, and examples
tests/                               Focused MedShiftLab-CXR tests
docs/medshiftlab/                    Research protocol, closeout, reproducibility, and packaging docs
docs/medshiftlab/templates/          Reusable documentation templates
```

## Documentation map

- [Research protocol](docs/medshiftlab/research_protocol.md)
- [Phase 11 final project closeout](docs/medshiftlab/final_project_closeout.md)
- [Reproducibility guide](docs/medshiftlab/reproducibility_guide.md)
- [CV-ready project summary](docs/medshiftlab/cv_project_summary.md)
- [Manuscript outline](docs/medshiftlab/manuscript_outline.md)
- [Experiment card template](docs/medshiftlab/templates/experiment_card_template.md)
- [Limitations](docs/medshiftlab/limitations.md)

## Suggested next steps

1. Keep this branch documentation-only and review the claim boundaries one more time before merge.
2. Tag a closeout snapshot after tests pass and docs are approved.
3. If real experiments are later authorized, create versioned local-only run directories and experiment cards for each run.
4. Treat any future full inference, internal evaluation, or external validation as a new explicitly approved phase.

## Real-data safe-subset status

A local/private Kaggle R1 run was completed on a 64-image frontal CheXpert validation subset using the TorchXRayVision DenseNet121 baseline adapter.

The run verified the real-image inference path after updating preprocessing to `phase5-baseline-inference-v2` with TorchXRayVision-compatible normalization.

Summary result at threshold 0.5:

- Mean AUROC: 0.7999
- Mean AUPRC: 0.4339
- Mean F1: 0.2385
- Mean sensitivity: 1.0
- Mean specificity: 0.0

This is an exploratory safe-subset smoke baseline only. It is not a full benchmark, not external validation, and not clinical validation.

See: [`docs/reports/chexpert_safe_subset_r1.md`](docs/reports/chexpert_safe_subset_r1.md)

## Threshold sweep status

A local/private R2 threshold sweep was completed on the same 64-image frontal CheXpert validation safe subset used in R1.

The sweep used thresholds from 0.00 to 1.00 in steps of 0.01 and selected per-label thresholds by maximizing F1 with deterministic tie-breakers.

Best exploratory thresholds:

- Atelectasis: 0.68
- Cardiomegaly: 0.63
- Pleural Effusion: 0.69
- Pneumonia: 0.65
- Pneumothorax: 0.64

This improved threshold behavior compared with the default threshold of 0.5, which produced all-positive predictions on the R1 subset. These thresholds are exploratory only and are not validated clinical or production operating points.

See: [`docs/reports/chexpert_threshold_sweep_r2.md`](docs/reports/chexpert_threshold_sweep_r2.md)

## Calibrated threshold evaluation status

A local/private R3 calibrated-threshold experiment was completed on 202 frontal CheXpert validation images.

Thresholds were selected on a deterministic calibration split and evaluated on a separate evaluation split.

Evaluation aggregate comparison:

- Default 0.5 mean F1: 0.3288
- Calibration-selected mean F1: 0.3442
- Default 0.5 mean specificity: 0.0000
- Calibration-selected mean specificity: 0.7874
- Default 0.5 mean balanced accuracy: 0.5000
- Calibration-selected mean balanced accuracy: 0.5899

This shows that calibration-selected thresholds avoid the all-positive behavior of the default 0.5 threshold, but sensitivity drops substantially for rare labels. These are exploratory thresholds only, not validated clinical or production operating points.

See: [`docs/reports/chexpert_calibrated_threshold_eval_r3.md`](docs/reports/chexpert_calibrated_threshold_eval_r3.md)

