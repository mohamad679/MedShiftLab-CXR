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

## Bootstrap uncertainty status

A local/private R4 bootstrap uncertainty summary was completed on the R3 evaluation split using 1000 bootstrap resamples.

Key exploratory bootstrap results:

- Calibration-selected mean F1: 0.3442, bootstrap 2.5%-97.5% range 0.2979-0.3802
- Default 0.5 mean F1: 0.3288, bootstrap 2.5%-97.5% range 0.2932-0.3608
- Calibration-selected mean specificity: 0.7874, bootstrap 2.5%-97.5% range 0.7303-0.8385
- Default 0.5 mean specificity: 0.0000, bootstrap 2.5%-97.5% range 0.0000-0.0000
- Calibration-selected mean balanced accuracy: 0.5899, bootstrap 2.5%-97.5% range 0.5570-0.6217
- Default 0.5 mean balanced accuracy: 0.5000, bootstrap 2.5%-97.5% range 0.4000-0.5000

This supports the conclusion that calibrated thresholds avoid all-positive behavior and improve specificity/balanced accuracy, while F1 differences remain uncertain. This is exploratory only and not clinical validation.

See: [`docs/reports/chexpert_bootstrap_uncertainty_r4.md`](docs/reports/chexpert_bootstrap_uncertainty_r4.md)

## Subgroup audit status

A local/private R5 subgroup audit was completed on the R3 evaluation split.

Subgroup variables:

- Sex
- View position
- Age bucket

The audit shows that calibration-selected thresholds avoid the default 0.5 all-positive behavior across evaluated slices and substantially improve specificity. However, subgroup sizes are small in several slices, especially PA view position, and rare-label estimates remain unstable.

This is exploratory only. It is not fairness validation, not clinical validation, not external validation, and not a full benchmark.

See: [`docs/reports/chexpert_subgroup_audit_r5.md`](docs/reports/chexpert_subgroup_audit_r5.md)

## Real-data evaluation closeout

The first CheXpert real-data evaluation sequence has been completed and documented across R1-R5:

- R1: real-image CheXpert safe-subset inference/evaluation
- R2: threshold sweep
- R3: calibrated threshold evaluation split
- R4: bootstrap uncertainty
- R5: subgroup/slice audit

The sequence verifies that MedShiftLab-CXR has a working real-image evaluation path with threshold calibration, uncertainty summaries, and subgroup audit tooling.

These results remain exploratory only. They are not clinical validation, not external validation, not fairness validation, and not a full benchmark.

Closeout summary: [`docs/reports/chexpert_realdata_evaluation_closeout_r6.md`](docs/reports/chexpert_realdata_evaluation_closeout_r6.md)

## Reproducible evaluation workflow status

A local evaluation workflow runner has been added to orchestrate the R2-R5 evaluation sequence:

- threshold sweep
- calibrated threshold evaluation
- bootstrap uncertainty
- subgroup audit

Script: `scripts/run_evaluation_workflow.py`

The runner supports dry-run command generation by default and full execution with `--execute`. It is intended for local/private exploratory evaluation outputs and should not be used to claim clinical validation, fairness validation, external validation, or full benchmarking.

Workflow report: [`docs/reports/chexpert_evaluation_workflow_r7.md`](docs/reports/chexpert_evaluation_workflow_r7.md)

## VinDr-CXR external validation scaffold

A VinDr-CXR external-validation scaffold has been added for the v0.4.0 track.

Added components:

- `configs/evaluation/vindr_cxr_label_mapping.json`
- `scripts/prepare_vindr_external_validation_inputs.py`
- `tests/test_medshiftlab_vindr_external_validation_scaffold.py`

This is a scaffold only. It prepares image-level labels, optional metadata, optional manifests, and local/private preparation summaries. It does not run VinDr-CXR inference and does not claim external validation, clinical validation, fairness validation, or full benchmarking.

Report: [`docs/reports/vindr_external_validation_scaffold_v040.md`](docs/reports/vindr_external_validation_scaffold_v040.md)


## VinDr/VinBigData metadata dry-run v0.4.1

v0.4.1 records a real VinDr/VinBigData metadata dry-run for external-validation preparation.

This stage verifies annotation-to-image matching and conservative label preparation only. It is not model training, not inference, and not external-validation metrics.

See `docs/reports/vindr_metadata_dryrun_v041.md`.

## VinDr/VinBigData reusable input workflow v0.4.2

v0.4.2 hardens the reusable VinDr/VinBigData external-validation input-preparation workflow.

The `scripts/prepare_vindr_external_validation_inputs.py` scaffold now fails fast when `--image-root` is invalid, when no manifest images are found, or when `--require-images` is enabled and any prepared image is missing. Manifest image paths remain relative to the provided image root.

This stage is input preparation only. It is not training, not inference, and not external-validation metrics.

See `docs/reports/vindr_external_validation_input_workflow_v042.md`.

## VinDr/VinBigData real input workflow run v0.4.3

v0.4.3 records a real Kaggle execution of the hardened VinDr/VinBigData input-preparation workflow.

The run produced local/private labels, manifest, and summary outputs for 15,000 matched images with zero missing images. These runtime outputs are not committed.

This stage is input preparation only. It is not training, not inference, and not external-validation metrics.

See `docs/reports/vindr_real_input_workflow_v043.md`.

## VinDr/VinBigData inference-readiness handoff v0.4.4

v0.4.4 records the inference-readiness handoff for VinDr/VinBigData external validation.

The handoff closes the input-preparation phase and defines the minimum safe requirements before any VinDr/VinBigData image inference is attempted.

This stage is not training, not inference, and not external-validation metrics.

See `docs/reports/vindr_inference_readiness_handoff_v044.md`.

## VinDr/VinBigData inference runner scaffold v0.4.5

v0.4.5 adds a VinDr/VinBigData inference-runner scaffold.

The scaffold validates prepared manifest and label inputs and writes a local/private inference-readiness summary. It does not load a model, generate predictions, run inference, or compute metrics.

See `docs/reports/vindr_inference_runner_scaffold_v045.md`.

## VinDr/VinBigData real inference scaffold run v0.4.6

v0.4.6 records a real Kaggle execution of the VinDr/VinBigData inference-runner scaffold.

The run consumed real prepared VinDr/VinBigData manifest and label inputs for 15,000 joinable samples. It did not load a model, generate predictions, run inference, or compute metrics.

See `docs/reports/vindr_real_inference_scaffold_run_v046.md`.

## VinDr/VinBigData prediction schema contract v0.4.7

v0.4.7 adds a prediction schema validator for future VinDr/VinBigData inference outputs.

The validator checks prediction CSV columns, probability ranges, duplicate sample IDs, and sample alignment against prepared manifest and label files. It does not generate predictions, run inference, or compute metrics.

See `docs/reports/vindr_prediction_schema_contract_v047.md`.

## VinDr/VinBigData prediction schema validation run v0.4.8

v0.4.8 records a controlled real Kaggle execution of the VinDr/VinBigData prediction-schema validator.

The run validates a local/private dummy prediction subset against real prepared VinDr/VinBigData manifest and label inputs. The dummy predictions were not generated by a model, and no metrics were computed.

See `docs/reports/vindr_prediction_schema_validation_run_v048.md`.

## VinDr/VinBigData real inference subset v0.5.0

v0.5.0 records the first real VinDr/VinBigData image inference run.

A pretrained TorchXRayVision DenseNet model was run on a 25-image real VinDr subset, producing raw predictions and mapped project-label predictions. The mapped prediction file passed schema validation in subset mode.

This stage confirms real inference execution. It does not compute external-validation metrics.

See `docs/reports/vindr_real_inference_subset_v050.md`.

## VinDr/VinBigData full real inference v0.5.1

v0.5.1 records the first full real VinDr/VinBigData image inference run.

A pretrained TorchXRayVision DenseNet model was run on all 15,000 prepared VinDr/VinBigData images. The mapped project-label prediction file contains 15,000 rows and passed prediction-schema validation with exact sample coverage.

This stage confirms full real inference execution. It does not compute external-validation metrics.

See `docs/reports/vindr_full_real_inference_v051.md`.

## VinDr/VinBigData external metrics v0.5.2

v0.5.2 records real external-validation metrics from the full VinDr/VinBigData inference run.

The evaluation used 15,000 real prediction rows from TorchXRayVision DenseNet121 and prepared VinDr/VinBigData labels. Metrics were computed for Atelectasis, Cardiomegaly, Pleural Effusion, and Pneumothorax. Pneumonia was excluded because VinBigData has zero positive Pneumonia labels under the conservative mapping policy.

This stage reports external dataset metrics only. It is not clinical validation and not a full benchmark.

See `docs/reports/vindr_external_metrics_v052.md`.

## VinDr/VinBigData external validation closeout v0.5.3

v0.5.3 closes the VinDr/VinBigData external-validation chain.

The completed chain includes real metadata matching, full real inference on 15,000 VinDr/VinBigData images, schema validation, and external metric computation for Atelectasis, Cardiomegaly, Pleural Effusion, and Pneumothorax.

This is external dataset evaluation, not clinical validation or deployment readiness.

See `docs/reports/vindr_external_validation_closeout_v053.md`.

## VinDr/VinBigData threshold analysis v0.6.0

v0.6.0 records threshold tuning and operating-point analysis for the VinDr/VinBigData external-validation predictions.

The analysis reused the full v0.5.1 prediction file and prepared labels for 15,000 samples. Best F1 thresholds were 0.55 for Atelectasis, 0.35 for Cardiomegaly, 0.50 for Pleural Effusion, and 0.50 for Pneumothorax.

This stage does not train a model and does not run new inference.

See `docs/reports/vindr_threshold_analysis_v060.md`.

## VinDr/VinBigData calibration analysis v0.6.1

v0.6.1 records calibration-oriented analysis for the VinDr/VinBigData external-validation predictions.

The analysis reused the full v0.5.1 prediction file and prepared labels for 15,000 samples. Cardiomegaly showed the best calibration, while Atelectasis and Pneumothorax showed substantial overprediction relative to observed prevalence.

This stage does not train a model and does not run new inference.

See `docs/reports/vindr_calibration_analysis_v061.md`.

## VinDr/VinBigData operating-point recommendation v0.6.2

v0.6.2 closes the VinDr/VinBigData threshold and calibration analysis phase.

Recommended reporting thresholds are 0.55 for Atelectasis, 0.35 for Cardiomegaly, 0.50 for Pleural Effusion, and 0.50 for Pneumothorax. Cardiomegaly and Pleural Effusion are the strongest labels, while Atelectasis and Pneumothorax require cautious interpretation due to weak precision and calibration behavior.

This stage does not train a model and does not run new inference.

See `docs/reports/vindr_operating_point_recommendation_v062.md`.

## CheXpert reference evaluation v0.7.0

v0.7.0 records a real CheXpert frontal validation reference evaluation.

The run evaluated 202 frontal CheXpert validation images with TorchXRayVision DenseNet121. At threshold 0.5, the model showed high sensitivity but low specificity, indicating substantial overprediction at the default operating point.

This stage does not train a model and does not establish clinical validation.

See `docs/reports/chexpert_reference_eval_v070.md`.
