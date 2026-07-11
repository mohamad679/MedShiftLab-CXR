# MedShiftLab-CXR

MedShiftLab-CXR is a reproducible research project for evaluating pretrained chest X-ray models under annotation uncertainty and cross-dataset shift. Release `v1.0.0` includes a real CheXpert reference evaluation and a completed VinDr/VinBigData external-dataset evaluation chain using TorchXRayVision DenseNet121. It remains a retrospective research evaluation, not a clinical system.

## Objective

The project supports controlled research around one question:

> How do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray models?

## Current release status

The authoritative current status is [`v1.0.0`](docs/reports/final_release_closeout_v100.md). Earlier phase, scaffold, and planning documents are retained as historical records and describe the repository at the time they were written.

## Cross-dataset bootstrap status

Cross-dataset bootstrap tooling is implemented, but real execution is blocked by
missing CheXpert sample-level artifacts. Existing historical aggregate metrics
remain unchanged, and no real cross-dataset bootstrap result is currently
claimed. See the [handoff report](docs/reports/chexpert_vindr_cross_dataset_bootstrap_handoff.md).

## Current implemented scope

- Conservative chest X-ray ontology and explicit `U-ignore`, `U-zero`, `U-one`, and `U-soft` label materialization
- Local/private dataset registry, path-containment checks, image loading, preprocessing, and prediction-schema validation
- Manual TorchXRayVision DenseNet121 inference workflow with standardized prediction exports
- Real CheXpert frontal reference evaluation on 202 validation images
- Full real VinDr/VinBigData inference on 15,000 prepared images
- VinDr/VinBigData external-dataset metrics for Atelectasis, Cardiomegaly, Pleural Effusion, and Pneumothorax
- Threshold, calibration-bin, bootstrap, subgroup, and failure-flag analysis utilities
- VinDr operating-point analysis and CheXpert/VinDr cross-dataset comparison
- JSON/CSV reporting exports and focused repository tests

## Explicit limitations

- The CheXpert result is a 202-image frontal validation reference evaluation, not a full CheXpert benchmark.
- The VinDr/VinBigData result is one retrospective external-dataset evaluation under a conservative four-label mapping; it is not prospective or clinical validation.
- Pneumonia is excluded from VinDr/VinBigData metrics because the conservative mapping produced zero positive Pneumonia labels.
- No MIMIC-CXR-JPG evaluation has been completed.
- No model training, fine-tuning, LoRA, PEFT, or new architecture is implemented.
- The four uncertainty strategies are materialized in the data layer, but their effects on separately fitted model heads and external generalization have not yet been compared.
- Calibration bins and scalar calibration metrics are implemented; calibration fitting and reliability-diagram rendering are not.
- Raw images, private manifests, predictions, model weights, and private runtime outputs are not committed.

## What is not claimed

This repository does not claim:

- clinical validation or diagnostic utility
- prospective validation
- fairness validation
- deployment or regulatory readiness
- state-of-the-art performance
- radiologist-level performance
- universal generalization beyond the evaluated datasets
- a comprehensive multi-model or multi-dataset clinical benchmark

## Safe setup

Create a Python 3.11 environment and install the package in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Install the optional TorchXRayVision dependencies only for explicitly authorized local inference:

```bash
pip install -e .[dev,torchxrayvision]
```

The repository does not include datasets, model weights, or private path configuration.

## Test command

```bash
bash scripts/run_medshiftlab_tests.sh
```

## Local/private data policy

- Keep raw medical data, restricted metadata, model weights, private predictions, and local outputs outside Git-tracked artifacts.
- Copy `configs/data/example_local_paths.yaml` to `configs/data/local_paths.yaml` locally and populate only the ignored copy.
- Use relative image paths inside manifests and label tables where the repository expects them.
- Do not commit private absolute paths, credentials, or access tokens.

## Manual workflow overview

1. Review the current release boundary in [`final_release_closeout_v100.md`](docs/reports/final_release_closeout_v100.md).
2. Create the ignored local data configuration from `configs/data/example_local_paths.yaml`.
3. Run the focused test suite before changing an experiment or report.
4. Reproduce or extend inference only in an authorized local/cloud environment with private data and outputs.
5. Evaluate standardized predictions with fixed label mappings, thresholds, calibration settings, and provenance.
6. Commit only code, configs, tests, and sanitized aggregate reports permitted by the dataset licenses.

Command examples are in [the reproducibility guide](docs/medshiftlab/reproducibility_guide.md).

## Repository structure

```text
src/medshiftlab/                     Core data, model-boundary, evaluation, experiment, and reporting modules
scripts/                             Manual/local CLI entry points
configs/                             Path-free tracked configs, protocol YAML, label mappings, and examples
tests/                               Focused MedShiftLab-CXR tests
docs/medshiftlab/                    Protocol, design, reproducibility, and historical closeout documents
docs/reports/                        Versioned sanitized run and release reports
```

## Documentation map

- [Current final release closeout v1.0.0](docs/reports/final_release_closeout_v100.md)
- [Current limitations](docs/medshiftlab/limitations.md)
- [Reproducibility guide](docs/medshiftlab/reproducibility_guide.md)
- [CV-ready project summary](docs/medshiftlab/cv_project_summary.md)
- [Research protocol — historical Phase 1 freeze](docs/medshiftlab/research_protocol.md)
- [Phase 11 scaffold closeout — historical](docs/medshiftlab/final_project_closeout.md)
- [Manuscript outline](docs/medshiftlab/manuscript_outline.md)
- [Experiment card template](docs/medshiftlab/templates/experiment_card_template.md)

## Compute-aware post-v1.0 priorities

1. Render calibration curves from the already exported calibration-bin tables.
2. Add bootstrap confidence intervals for VinDr metrics and CheXpert-to-VinDr performance deltas without new inference.
3. Compare `U-ignore`, `U-zero`, `U-one`, and `U-soft` against the existing fixed prediction artifacts.
4. Add a patient-level prevalence-aware subset sampler before any new development experiment.
5. If model refitting is required, extract frozen DenseNet121 embeddings once and train lightweight linear probes on CPU.

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

## CheXpert vs VinDr cross-dataset comparison v0.7.1

v0.7.1 compares the CheXpert frontal validation reference evaluation against the VinDr/VinBigData external-validation evaluation.

The comparison shows that Cardiomegaly and Pleural Effusion are the most stable labels, while Atelectasis and Pneumothorax remain difficult under prevalence shift, AUPRC/F1 behavior, and threshold sensitivity.

This stage does not train a model and does not run new inference.

See `docs/reports/chexpert_vindr_cross_dataset_comparison_v071.md`.

## CheXpert/VinDr reference comparison closeout v0.7.2

v0.7.2 closes the CheXpert/VinDr reference-comparison phase.

The project now has a real CheXpert reference evaluation, full VinDr/VinBigData external inference, VinDr/VinBigData external metrics, threshold analysis, calibration analysis, and a cross-dataset comparison. This supports a credible cross-dataset shift narrative without claiming clinical validation.

This stage does not train a model and does not run new inference.

See `docs/reports/chexpert_vindr_reference_comparison_closeout_v072.md`.

## Final release v1.0.0

MedShiftLab-CXR v1.0.0 closes the current project scope.

The project now includes real CheXpert reference evaluation, full VinDr/VinBigData external inference, VinDr/VinBigData external metrics, threshold analysis, calibration analysis, and cross-dataset comparison.

This release supports a dataset-shift evaluation narrative. It does not claim clinical validation, diagnostic deployment readiness, prospective validation, fairness validation, SOTA performance, or regulatory readiness.

See `docs/reports/final_release_closeout_v100.md`.
