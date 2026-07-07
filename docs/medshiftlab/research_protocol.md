# MedShiftLab-CXR Research Protocol

## Protocol status

This document is the Phase 1 protocol freeze for MedShiftLab-CXR. It defines the intended study before formal external validation. Items described as planned are protocol commitments, not completed experiments or results.

The tracked `chexpert_small_frontal1000_torchxrayvision` artifacts document a prior, pre-freeze subset execution through standalone scripts. They are smoke/subset artifacts only. They are not evidence of a completed benchmark, external validation, clinical validation, or integrated package-level inference.

## 1. Project objective

MedShiftLab-CXR is a reproducible, data-centric framework for studying how annotation uncertainty, dataset curation, calibration, and cross-dataset shift affect pretrained chest X-ray classification models. The primary task is multi-label classification over a conservative, explicitly harmonized label set.

The project evaluates fixed pretrained models and explicit data/evaluation policies. It does not aim to invent a new architecture, train a foundation model from scratch, or support clinical decision-making.

Primary research question:

> How do annotation uncertainty, dataset curation choices, and cross-dataset distribution shift influence the robustness, calibration, and failure modes of pretrained chest X-ray models?

## 2. Dataset roles and cohorts

### 2.1 Development cohort: CheXpert

CheXpert is the development and internal-protocol dataset. Subject to authorized local access, it may be used for:

- metadata and image-quality audit;
- label harmonization and uncertainty-strategy comparison;
- development split construction;
- model and preprocessing smoke checks;
- threshold selection and optional calibration fitting on the development validation split only;
- internal evaluation on a held-out internal test split.

All CheXpert development, validation, and internal-test partitions must be patient-disjoint when patient identifiers are available. The internal test split must not be used to choose preprocessing, labels, uncertainty policy, thresholds, calibration parameters, models, or hyperparameters.

### 2.2 Internal validation

Internal validation means evaluation on a held-out, patient-disjoint CheXpert internal test split after all choices have been made using only the CheXpert development train/validation partitions.

The split manifest, inclusion and exclusion rules, model identifier, preprocessing version, label mapping, uncertainty policies, fixed threshold policy, optional fitted calibrator, ECE binning, and bootstrap procedure must be recorded before internal-test evaluation. Any change prompted by internal-test results creates a new protocol version and requires a new untouched internal-test split where feasible.

### 2.3 External validation candidates

MIMIC-CXR-JPG and VinDr-CXR are candidate external validation datasets. The final external dataset may be one or both, depending on authorized access, label compatibility, metadata availability, and licensing constraints. Candidate status does not imply that external validation has been completed.

External datasets may be used only after the protocol and candidate-specific label mapping are frozen. They must not be used for:

- training or fine-tuning;
- threshold or calibration fitting;
- hyperparameter tuning;
- model, checkpoint, preprocessing, or uncertainty-strategy selection;
- inclusion-rule optimization based on outcomes;
- protocol editing in response to external performance.

External evaluation uses the model, preprocessing, thresholds, calibrator if any, labels, metrics, and analysis rules fixed from CheXpert development. Dataset-specific technical adaptations needed to read files or map already-prespecified labels must be documented without using external outcome results to optimize the protocol.

## 3. Label harmonization

The current conservative core labels are:

- Atelectasis
- Cardiomegaly
- Pleural Effusion
- Pneumonia
- Pneumothorax

`No Finding` is analyzed separately and is not treated as a pathology label. The current CheXpert/VinDr-CXR mapping source is `configs/labels/cxr_common_labels.yaml`.

Before evaluating any external candidate, its source labels must be mapped explicitly to the project ontology. Mapping decisions must be based on source definitions and annotation procedures, not observed model performance. Unsupported or materially non-equivalent labels must be excluded and documented. The project does not assume that identically named labels are clinically or operationally equivalent across datasets.

Any MIMIC-CXR-JPG mapping is future work and must be reviewed and frozen before that dataset is evaluated. Label additions, removals, or semantic changes after external results are inspected require a separately versioned protocol and cannot be presented as confirmatory evaluation under this freeze.

## 4. CheXpert uncertainty strategies

CheXpert uncertain labels (`-1`) are evaluated under four explicit conditions:

- **U-ignore:** omit the uncertain target for that sample and label.
- **U-zero:** map the uncertain target to `0`.
- **U-one:** map the uncertain target to `1`.
- **U-soft:** map the uncertain target to `0.5` unless a later protocol version prespecifies another value before evaluation.

Missing labels remain missing. These strategies are modeling and evaluation policies, not clinical truth. Results must retain the strategy identifier. Binary discrimination and threshold metrics use targets equal to `0` or `1`; soft targets may contribute only to metrics that explicitly support probabilistic targets, such as Brier score and ECE.

External datasets are evaluated according to their documented labels. CheXpert uncertainty transformations must not be imposed on an external dataset unless its annotation schema contains a justified, prespecified equivalent.

## 5. Primary metrics and uncertainty estimates

The following are primary, label-wise metrics or outputs:

- AUROC;
- AUPRC;
- F1;
- sensitivity;
- specificity;
- Brier score;
- Expected Calibration Error (ECE);
- calibration curve;
- bootstrap 95% confidence intervals.

F1, sensitivity, and specificity are primarily reported at a fixed threshold of `0.5`. A secondary label-specific threshold may be selected using CheXpert validation only, provided its selection rule and value are recorded before internal-test and external evaluation. No threshold may be selected or revised on an internal-test or external dataset.

AUROC and AUPRC are reported only when both binary classes are represented. Every metric must include its available sample count and applicable positive/negative counts. Undefined metrics remain undefined rather than being imputed.

Confidence intervals use a prespecified percentile bootstrap with 2,000 replicates. Resampling is performed at patient level when patient identifiers are available so that all images from one patient remain together. If patient identifiers are unavailable, image-level resampling may be used only with that limitation stated. The random seed, software version, and resampling unit must be recorded. This bootstrap procedure is planned and is not yet implemented by the current evaluation package.

## 6. Calibration analysis

Calibration analysis includes label-wise Brier score, ECE, and calibration curves. ECE and calibration curves use 10 equal-width probability bins over `[0, 1]` unless a future protocol version is frozen before evaluation. Empty-bin handling and bin counts must be recorded.

If calibration fitting is performed, the method and parameters must be fitted using CheXpert validation only. The fitted calibrator must then be frozen and applied unchanged to CheXpert internal test and external datasets. Calibration must never be refitted on internal-test or external outcomes.

Calibration results must be reported separately by dataset and label. They must not be described as clinical confidence or deployment readiness. The tracked prior subset artifacts contain aggregate Brier/ECE values, but they do not constitute the complete frozen calibration analysis defined here.

## 7. Subgroup analysis

Where the required metadata exist and use is permitted, results are stratified by:

- sex, using source-dataset categories without inferring missing identity;
- age group: `<40`, `40–64`, `>=65`, and unknown;
- view position: AP, PA, lateral, other, and unknown;
- dataset source;
- label uncertainty strategy.

Subgroup reports must include support counts and class counts. Metrics requiring both classes are not reported for single-class subgroups. Missing attributes remain unknown rather than being inferred. Small or sparse groups must be identified, and subgroup comparisons must not be presented as causal, clinical, fairness, or population-generalization conclusions.

## 8. Leakage-control rules

The following controls are mandatory:

1. Split by patient, not image, whenever patient identity is available.
2. Deduplicate patients and images across development, validation, internal-test, and external cohorts where identifiers or safe hashes permit.
3. Fit thresholds, calibrators, linear probes, and any learned preprocessing on CheXpert development data only.
4. Keep CheXpert internal test unavailable for model and protocol selection.
5. Keep external labels and results unavailable until the external validation freeze is recorded.
6. Do not select labels, models, checkpoints, subgroups, exclusions, or metrics based on internal-test or external performance.
7. Record all exclusions and failed samples; do not silently remove inference failures.
8. Do not transfer restricted images, identifiers, private paths, or protected metadata into Git artifacts.

Any known or suspected leakage invalidates the affected evaluation until the split and outputs are regenerated under a new recorded run.

## 9. External validation freeze rule

Before computing performance on an external candidate, the following must be committed together as a versioned freeze:

- protocol version and Git commit;
- external candidate and cohort definition;
- patient/image inclusion and exclusion rules;
- split and deduplication rules;
- label harmonization table;
- uncertainty policy;
- model, weights identifier, and preprocessing specification;
- thresholds and any CheXpert-fitted calibrator;
- primary metrics, calibration bins, bootstrap procedure, and subgroup rules;
- artifact schema and planned output locations.

After external labels or performance results are inspected, none of these items may be changed for that confirmatory run. Corrections required for a software defect or invalid input must be documented, versioned, and rerun as a new analysis; both the reason and impact must be retained. Exploratory post-hoc analyses must be labeled exploratory and must not replace the frozen result.

## 10. Artifact and versioning rules

Each run must record, where applicable:

- protocol version, run identifier, timestamp, and Git commit;
- dataset name, authorized source/version, cohort manifest checksum, and split name;
- model and weights identifier, dependency versions, device, and preprocessing version;
- label ontology version, uncertainty strategy, thresholds, calibration settings, and random seeds;
- sample counts, exclusions, failures, and output schema version.

Raw medical images, restricted metadata, credentials, model weights, and predictions containing private absolute paths must not be committed. Local manifests and prediction tables remain outside Git unless licensing and privacy review explicitly permit a de-identified derivative. Commit-eligible artifacts should be aggregate, non-identifying outputs with provenance and limitations. Artifacts must not be silently overwritten; a changed protocol or input produces a new versioned run directory.

Historical artifacts must retain their original context. The tracked frontal-1000 subset artifacts are pre-freeze standalone-script outputs and must remain distinguishable from future frozen internal or external evaluations.

## 11. Allowed and disallowed claims

Allowed claims, when directly supported by repository contents:

- MedShiftLab-CXR is a reproducible data-centric research scaffold.
- The repository implements label ontology, uncertainty handling, metadata contracts, evaluation schemas and metrics, prediction contracts, experiment runners, and report export.
- A tracked prior CheXpert frontal-1000 subset artifact set documents standalone TorchXRayVision execution under its stated limitations.
- The protocol intends CheXpert for development/internal evaluation and MIMIC-CXR-JPG and/or VinDr-CXR as external validation candidates.

Disallowed claims:

- clinical validation, diagnostic utility, safety, or deployment readiness;
- state-of-the-art, radiologist-level, or model-superiority claims;
- completed benchmark or completed external validation claims;
- generalization to hospitals, populations, devices, or workflows not evaluated under a frozen protocol;
- perfect equivalence of labels across datasets;
- completed benchmark-grade adapter integration, full-dataset baseline inference, or any inference result presented as confirmatory evaluation without the frozen protocol steps above;
- model training or fine-tuning that did not occur;
- interpreting uncertainty transformations as clinical ground truth.

## 12. Phase decisions

- **Phase 1:** freeze this research protocol and reconcile documentation status.
- **Current inference boundary:** `TorchXRayVisionAdapter.predict_records()` and `scripts/run_baseline_inference.py` provide a manual-only local baseline inference path for small subsets. They require authorized local data, explicit local configuration, optional dependencies, and an externally initialized or explicitly authorized local model-init step. They do not establish benchmark completion, external validation, or clinical validation.
- **Phase 2:** add local/private data configuration and a dataset registry without committing private paths or data.
- **Phase 3:** add reusable package-level image loading and preprocessing.
- **Phase 4:** standardize the prediction output schema and adapter interface before adding real model adapters.
- **Phase 5:** integrate a safe baseline adapter inference path on top of the reusable loading and standardized prediction layers.
- **Phase 6:** add evaluation orchestration for standardized prediction files and manual-only run-control around the baseline inference path without broadening it into a full-dataset default.
- **Current prediction-evaluation boundary:** local standardized prediction JSON/CSV files may be evaluated against a matching local label CSV through `scripts/run_real_prediction_evaluation.py`. Bootstrap confidence intervals and calibration-figure export remain deferred until a later phase.
- **Phase 7:** add CheXpert internal protocol scaffolding only: patient-level split manifests, explicit uncertainty-strategy label-table materialization, reproducible protocol configs, and manual-only preparation through `scripts/prepare_chexpert_internal_protocol.py`.
- **Current internal-protocol boundary:** the Phase 7 utilities prepare local/private CheXpert split and label artifacts that are compatible with the Phase 6 evaluation path once real local prediction files exist. They do not themselves run full CheXpert inference, generate completed internal benchmark results, or change the prohibition on benchmark, external-validation, clinical-validation, or SOTA claims.
- **Phase 8:** add external-validation setup scaffolding only: dataset-specific external protocol configs, label harmonization configs, external manifest validation, optional internal/external patient-overlap checks, and manual-only preparation through `scripts/prepare_external_validation_protocol.py`.
- **Current external-validation boundary:** the Phase 8 utilities prepare local/private MIMIC-CXR-JPG and VinDr-CXR manifest and label-table artifacts that are compatible with the Phase 6 evaluation path once real local external prediction files exist. They do not themselves run real external validation, inspect external outcomes, tune thresholds or hyperparameters on external data, or alter the prohibition on benchmark, external-validation, clinical-validation, training, or SOTA claims.
- **Phase 9:** add a path-free adapter-candidate registry, safe adapter factory, manual-only configuration profiles, and one mock-compatible generic foundation-model scaffold. Registry inspection and mock adapter construction do not initialize optional dependencies or resolve weights.
- **Current foundation-model boundary:** the generic Phase 9 scaffold accepts injected fake backends for contract tests only. Real initialization is refused unless explicitly authorized, missing optional dependencies fail clearly, and even an authorized environment stops with `NotImplementedError` because no real foundation-model backend was integrated. TorchXRayVision remains the only manual optional baseline path. No training, fine-tuning, LoRA/PEFT, full-dataset inference, benchmark completion, external-validation completion, clinical validation, or performance claim is introduced.

The dataset registry, reusable package-level image loading, prediction schema standardization, and bounded baseline adapter inference were deferred by Phase 1 and implemented incrementally in later phases under the boundaries above.

## 13. Protocol deviations

Any deviation must identify the affected rule, rationale, date, Git commit, and whether outcomes had already been inspected. Deviations made after internal-test or external outcomes are known are exploratory unless a new independent evaluation cohort is reserved and a new protocol is frozen before its use.
