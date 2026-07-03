# MedShiftLab-CXR Evaluation Layer Design

## Purpose

The evaluation layer provides a small, reproducible interface for evaluating pretrained chest X-ray model scores under annotation uncertainty and cross-dataset distribution shift. It separates metric computation and report construction from data access, image processing, and model execution.

## Implemented Scope

For each requested label, the layer computes:

- AUROC
- area under the precision-recall curve (AUPRC)
- Brier score
- expected calibration error (ECE)
- F1 score
- sensitivity
- specificity

`EvaluationReport` records dataset and model identity, split, uncertainty strategy, threshold, ECE bin count, optional notes, aggregate metrics, and label-wise metrics. The table-to-report interface converts row-based prediction tables into label-wise target and score mappings before constructing this report.

## Label Handling

- `None` and other recognized missing target or score values are excluded from metric computation.
- Soft labels in the interval `[0, 1]` are retained for Brier score and ECE.
- AUROC, AUPRC, F1, sensitivity, and specificity use only targets equal to binary `0` or `1`.
- The evaluation layer does not impute labels or reinterpret uncertainty. The selected uncertainty strategy must be recorded in report metadata.

This separation prevents soft annotation strategies from being silently treated as hard diagnostic labels while still allowing their probabilistic information to contribute to calibration-oriented metrics.

## Table Interface

Each input row is a mapping with one target and one prediction score column per requested label:

- `true_<label_name>`
- `score_<label_name>`

For example, `true_Atelectasis` and `score_Atelectasis` form one label pair. Every requested pair must be present in every row. The table adapter preserves values without coercion; missing-value filtering and numeric validation occur in the metric layer. Custom target and score prefixes are supported when explicitly supplied.

## Explicit Limitations

The evaluation layer provides:

- no model inference
- no training
- no threshold tuning
- no calibration fitting
- no dataset loading
- no image loading

Thresholds and ECE bin counts are evaluation inputs, not learned parameters. Dataset loading, prediction generation, and artifact export remain separate responsibilities.

## Next Dependencies

Stage 8, the model adapter layer, will provide prediction scores from supported pretrained chest X-ray foundation models. Stage 9, the experiment pipeline, will connect data records, prediction scores, evaluation reports, and file exports into reproducible experiment runs.
