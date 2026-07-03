# MedShiftLab-CXR Model Adapter Layer Design

## Purpose

The model adapter layer separates pretrained chest X-ray model outputs from evaluation logic. Stable `PredictionRecord` and `PredictionBatch` schemas make prediction identity, model identity, label coverage, and score validation explicit. Future pretrained CXR adapters can therefore produce the same contract without changing evaluation code.

## Implemented Scope

The layer provides:

- `PredictionRecord` for one image's validated label-wise probabilities and provenance
- `PredictionBatch` for a non-empty, single-model collection with explicit label coverage
- `build_score_mapping_from_predictions` for ordered label-wise score mappings
- `build_prediction_table_rows` for `score_<label_name>` table columns
- the `CXRModelAdapter` protocol for stable adapter properties and prediction behavior
- `MockCXRModelAdapter` for deterministic tests and pipeline fixtures
- an optional `TorchXRayVisionAdapter` boundary for mapping precomputed output columns
- a prediction-to-evaluation bridge for combining validated data records and prediction batches

These interfaces preserve record order where specified and validate probabilities before evaluation. The mock adapter is a fixture, not a scientific model.

## Dependency Safety

`torch` and `torchxrayvision` are optional and are not required by the test suite. `torchxrayvision` is not imported when `medshiftlab.models` is imported; availability is checked without eagerly loading either package.

The optional TorchXRayVision adapter does not construct a model, download weights, preprocess images, or execute inference. It currently maps externally supplied, row-like model outputs to project labels and returns the standard prediction schema. This is an integration boundary rather than a complete inference implementation.

## Prediction-to-Evaluation Bridge

Data records and prediction records are matched by `image_id`, not by position. The bridge:

- rejects duplicate `image_id` values in either input
- rejects missing predictions for data records
- rejects predictions whose `image_id` is unknown to the data records
- validates requested target and score label coverage
- preserves input data-record order
- emits `true_<label_name>` and `score_<label_name>` columns

The bridge does not coerce targets or scores. It delegates report construction and metric behavior to the evaluation layer.

## Explicit Limitations

The model adapter layer provides:

- no real image loading
- no real model inference
- no model weight download
- no training
- no threshold tuning
- no calibration fitting
- no dataset downloading
- no frontend or API integration

## Next Dependency

Stage 9, the experiment pipeline, will connect data records, adapter predictions, evaluation reports, and file exports into reproducible experiment runs.
