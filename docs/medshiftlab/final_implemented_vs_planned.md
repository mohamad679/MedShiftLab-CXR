# MedShiftLab-CXR

Data-centric evaluation framework for pretrained chest X-ray foundation models under annotation uncertainty and cross-dataset distribution shift.

## Implemented Components

### Research Protocol and Scope

- The research protocol defines CheXpert internal evaluation, VinDr-CXR external validation, annotation-uncertainty analysis, distribution-shift analysis, and conservative scientific claims.
- Dataset cards, a pretrained-model card, limitations, roadmap, and layer-specific design documents record intended use, boundaries, and future work.
- The README identifies MedShiftLab-CXR as the active research track while retaining NeuroSight only as legacy context.

### Label and Data Layer

- A versioned label ontology defines the core CXR labels and conservative CheXpert/VinDr-CXR mappings.
- CheXpert uncertainty strategies implement U-ignore, U-zero, U-one, and U-soft transformations without treating any strategy as clinical truth.
- The CheXpert schema parses metadata rows into validated records; the CSV loader reads metadata only and does not load images.
- The VinDr-CXR schema parses image-level metadata and binary labels into validated records; raw annotation aggregation is not implemented.
- Dataset summary utilities report record, patient, missing-label, prevalence, and target-count statistics from validated records.
- A registry defines CheXpert, MIMIC-CXR-JPG, and VinDr-CXR roles and required local path fields. A tracked null-only template is separated from the ignored real local configuration.

### Evaluation Layer

- Label-wise evaluation computes AUROC, AUPRC, Brier score, expected calibration error, F1, sensitivity, and specificity from supplied targets and scores.
- Missing targets and scores are excluded. Soft targets contribute to Brier score and ECE, while discrimination and threshold metrics use binary targets only.
- `EvaluationReport` stores dataset/model identity, split, uncertainty strategy, evaluation settings, aggregate metrics, and label-wise metrics.
- The table interface converts explicit `true_<label>` and `score_<label>` columns into evaluation reports. Thresholds are inputs; they are not tuned.

### Model Adapter Layer

- `PredictionRecord` and `PredictionBatch` validate image identity, model identity, label coverage, and probability ranges.
- `CXRModelAdapter` defines the model-independent prediction contract, and `MockCXRModelAdapter` provides deterministic test predictions.
- The optional `TorchXRayVisionAdapter` maps externally supplied output columns into the standard prediction schema. It does not preprocess images, construct models, download weights, or run inference.
- Separate standalone scripts provide a bounded image-loading and TorchXRayVision smoke/subset execution path. Tracked aggregate artifacts document one prior 1,000-image frontal CheXpert subset run.
- The prediction-to-evaluation bridge joins data records and predictions by `image_id`, validates coverage and duplicates, and builds evaluation rows.

### Experiment and Reporting Layer

- JSON export writes the complete structured evaluation report; CSV export writes one stable row per evaluated label.
- The in-memory experiment runner connects validated records, an adapter, predictions, evaluation rows, and an `EvaluationReport`.
- The file-exporting runner delegates evaluation to the in-memory runner and writes the JSON/CSV report bundle.
- These runners operate on supplied records and adapter outputs; they do not access datasets, images, or model weights.

### Repository and Test Boundary

- `scripts/run_medshiftlab_tests.sh` runs only the focused MedShiftLab-CXR tests and excludes the legacy NeuroSight pytest configuration.
- `docs/medshiftlab/ci_test_boundary.md` defines the authoritative coverage and exclusions.
- Remote CI is intentionally deferred because the current installation path is not yet separated from legacy, torch-heavy dependencies.

## Planned or Future Components

- reusable package-level image loading and preprocessing
- real TorchXRayVision inference integration through the adapter interface
- CheXpert internal evaluation using authorized actual metadata and images
- MIMIC-CXR-JPG and/or VinDr-CXR external validation under the frozen protocol
- optional robustness stress tests
- optional calibration fitting using CheXpert validation data only, without external-test leakage
- optional richer comparison reports and figures
- future CT/MRI extension only after the CXR framework is validated

## Explicit Non-Claims

The project currently does not claim:

- clinical validity
- diagnostic deployment readiness
- completed benchmark or external validation results
- fully integrated package-level inference
- training a new foundation model
- state-of-the-art performance
- FDA, CE, or other regulatory readiness

## PhD Relevance

The implemented framework makes data and evaluation choices explicit rather than treating model architecture as the only research variable. It is relevant to data-centric AI through validated label, metadata, and uncertainty contracts; to radiological imaging through a focused chest X-ray protocol; and to annotation uncertainty through auditable CheXpert label strategies. Its separation of internal and external datasets supports distribution-shift research. Current discrimination and calibration metrics, together with the extensible experiment boundary, provide a controlled basis for future robustness evaluation. Explicit provenance, reproducible reports, leakage-aware boundaries, and conservative claims align the work with trustworthy AI research.

## Reproducibility

```bash
bash scripts/run_medshiftlab_tests.sh
```
