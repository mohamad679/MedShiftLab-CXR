# MedShiftLab-CXR Experiment Pipeline Design

## Purpose

The experiment pipeline connects validated chest X-ray records, adapter predictions, evaluation reports, and file exports through explicit, testable contracts. It keeps each run reproducible and inspectable while avoiding any coupling between evaluation logic and a specific pretrained model implementation.

## Implemented Scope

The pipeline provides:

- complete `EvaluationReport` JSON export
- stable label-metrics CSV export
- combined JSON/CSV report bundle export
- `InMemoryExperimentConfig` for dataset, label, split, uncertainty, and metric settings
- `InMemoryExperimentResult` for predictions, evaluation rows, and the report
- `run_in_memory_evaluation_experiment` for prediction and evaluation without file writes
- `ExportedExperimentResult` for the in-memory result and written artifact paths
- `run_and_export_evaluation_experiment` for report-bundle creation

The exported runner delegates computation to the in-memory runner and serialization to the reporting layer. It does not duplicate evaluation logic or write raw records and predictions.

## Current Pipeline Flow

```text
validated records
    -> adapter.predict_records()
    -> PredictionBatch
    -> evaluation rows
    -> EvaluationReport
    -> JSON/CSV report bundle
```

Configuration records the dataset name, requested labels, split, uncertainty strategy, threshold, ECE bin count, and optional notes. Record identity is preserved through `image_id`, and the evaluation bridge matches records to predictions by that identifier.

## Output Artifacts

- `evaluation_report.json` contains the complete structured `EvaluationReport`: run metadata, aggregate metrics, and label-wise metrics.
- `evaluation_label_metrics.csv` contains one stable, flat row per evaluated label for comparison and downstream analysis.

The current bundle intentionally excludes prediction tables, dataset records, images, and model files.

## Scientific Boundaries

The experiment pipeline provides:

- no real model inference
- no image loading
- no dataset downloading
- no training
- no threshold tuning
- no calibration fitting
- no model-weight downloading
- no frontend or API integration

Thresholds and uncertainty strategies are explicit experiment inputs. Their selection is not performed by this pipeline.

## Relevance to the PhD Application

This design demonstrates a data-centric evaluation framework rather than a model-training showcase. It separates dataset handling, model prediction, evaluation, and reporting so that each stage can be audited and replaced independently. The same contracts can support future CheXpert internal evaluation and strict VinDr-CXR external validation without changing metric or export logic.

## Next Dependencies

Stage 10 will focus on repository positioning, the README, and a minimal CI/test boundary. Stage 11 will produce the final PhD application package and interview narrative.
