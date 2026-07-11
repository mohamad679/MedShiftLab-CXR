# MedShiftLab-CXR Reproducibility Guide

## Scope

This guide covers repository-level reproducibility and safe manual/local commands for MedShiftLab-CXR. The repository records completed CheXpert reference and VinDr/VinBigData external-dataset evaluations, but the commands in this guide do not by themselves reproduce private runtime artifacts or authorize new data access, training, or clinical use.

## Environment setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

If and only if you intentionally plan to run the manual optional TorchXRayVision baseline path:

```bash
pip install -e .[dev,torchxrayvision]
```

## Running tests

Focused repository verification:

```bash
bash scripts/run_medshiftlab_tests.sh
```

Optional lightweight syntax check:

```bash
python -m compileall src scripts tests
```

## Local/private config expectations

1. Copy `configs/data/example_local_paths.yaml` to `configs/data/local_paths.yaml`.
2. Populate only the ignored `configs/data/local_paths.yaml`.
3. Keep dataset roots, metadata files, model weights, predictions, and evaluation outputs out of Git.
4. Use placeholders in documentation and notes instead of private absolute paths.

Tracked example:

```bash
cp configs/data/example_local_paths.yaml configs/data/local_paths.yaml
```

## Safe manual commands

All commands below use repository-relative paths or placeholders such as `<PRIVATE_METADATA_CSV>`. Replace placeholders only in your private local environment.

### 1. Validate local/private dataset registry entries

This checks that a dataset is configured and that required path fields are present without printing the private path values:

```bash
PYTHONPATH=src python -c "from medshiftlab.data import load_local_data_config, require_local_dataset_paths; cfg=load_local_data_config('configs/data/local_paths.yaml'); paths=require_local_dataset_paths(cfg, 'chexpert'); print({'dataset':'chexpert','validated_fields':sorted(paths)})"
```

Repeat with `mimic_cxr_jpg` or `vindr_cxr` as needed.

### 2. Image loading smoke test

```bash
PYTHONPATH=src python scripts/run_image_loading_smoke_test.py \
  --config configs/data/local_paths.yaml \
  --dataset chexpert \
  --limit 5 \
  --output-mode grayscale \
  --height 224 \
  --width 224 \
  --normalization zero_one
```

### 3. Manual bounded baseline inference

Requirements:

- optional `torchxrayvision` dependencies installed
- authorized local data available
- a private manifest CSV with relative `image_path` values
- explicit opt-in to model initialization

Example:

```bash
PYTHONPATH=src python scripts/run_baseline_inference.py \
  --config configs/data/local_paths.yaml \
  --dataset chexpert \
  --manifest-csv <PRIVATE_MANIFEST_CSV> \
  --output-json outputs/local_predictions/<RUN_ID>/predictions.json \
  --output-csv outputs/local_predictions/<RUN_ID>/predictions.csv \
  --limit 16 \
  --allow-model-init \
  --device cpu \
  --batch-size 4
```

Notes:

- The script is bounded by a safe default and rejects larger runs unless `--allow-large-run` is passed.
- This command is manual/local only and does not establish benchmark completion or external validation.

### 4. Standardized prediction evaluation

```bash
PYTHONPATH=src python scripts/run_real_prediction_evaluation.py \
  --predictions <PRIVATE_PREDICTIONS_JSON_OR_CSV> \
  --labels-csv <PRIVATE_LABEL_TABLE_CSV> \
  --output-json outputs/local_evaluations/<RUN_ID>/evaluation_report.json \
  --output-csv outputs/local_evaluations/<RUN_ID>/label_metrics.csv \
  --threshold 0.5 \
  --n-bins 10 \
  --limit 256
```

Keep `--bootstrap-iters` at `0` for this CLI path unless the implementation changes; bootstrap analysis is handled through the robustness-analysis flow.

### 5. CheXpert internal protocol preparation

```bash
PYTHONPATH=src python scripts/prepare_chexpert_internal_protocol.py \
  --metadata-csv <PRIVATE_CHEXPERT_METADATA_CSV> \
  --output-dir results/local_private_runs/<RUN_ID>/chexpert_internal_protocol \
  --uncertainty-strategy all \
  --seed 2026 \
  --split-names train,validation,test \
  --split-fractions 0.70,0.15,0.15 \
  --limit 256 \
  --write-split-manifest \
  --write-label-tables
```

This prepares manifests and label tables. It does not run inference.

### 6. External-validation setup preparation

Example for MIMIC-CXR-JPG:

```bash
PYTHONPATH=src python scripts/prepare_external_validation_protocol.py \
  --dataset mimic_cxr_jpg \
  --metadata-csv <PRIVATE_MIMIC_METADATA_CSV> \
  --output-dir results/local_private_runs/<RUN_ID>/mimic_cxr_jpg_external_validation \
  --internal-manifest <PRIVATE_CHEXPERT_INTERNAL_MANIFEST_CSV> \
  --limit 256 \
  --write-manifest \
  --write-label-table
```

Example for VinDr-CXR:

```bash
PYTHONPATH=src python scripts/prepare_external_validation_protocol.py \
  --dataset vindr_cxr \
  --metadata-csv <PRIVATE_VINDR_METADATA_CSV> \
  --output-dir results/local_private_runs/<RUN_ID>/vindr_cxr_external_validation \
  --internal-manifest <PRIVATE_CHEXPERT_INTERNAL_MANIFEST_CSV> \
  --limit 256 \
  --write-manifest \
  --write-label-table
```

These commands validate preparation artifacts only. They do not run external inference or evaluation.

### 7. Robustness, calibration, subgroup, and bootstrap analysis

Default bounded example:

```bash
PYTHONPATH=src python scripts/run_robustness_calibration_analysis.py \
  --predictions <PRIVATE_PREDICTIONS_JSON_OR_CSV> \
  --labels-csv <PRIVATE_LABEL_TABLE_CSV> \
  --metadata-csv <PRIVATE_SUBGROUP_METADATA_CSV> \
  --baseline-evaluation-json <PRIVATE_BASELINE_EVALUATION_JSON> \
  --output-dir outputs/local_robustness/<RUN_ID> \
  --export-calibration-csv \
  --bootstrap-iters 0 \
  --bootstrap-metrics auroc,brier_score,ece \
  --seed 2026 \
  --confidence-level 0.95 \
  --subgroup-columns dataset_name split uncertainty_strategy sex age_group view_position \
  --minimum-subgroup-size 20 \
  --threshold 0.5 \
  --n-bins 10 \
  --limit 256
```

Optional aggregate reliability-curve PNG export:

```bash
PYTHONPATH=src python scripts/run_robustness_calibration_analysis.py \
  --predictions <PRIVATE_PREDICTIONS_JSON_OR_CSV> \
  --labels-csv <PRIVATE_LABEL_TABLE_CSV> \
  --output-dir outputs/local_robustness/<RUN_ID> \
  --export-calibration-plot
```

The `--export-calibration-plot` flag writes `calibration_curves.png` from the
existing aggregate calibration bins in the robustness report. It does not run
new inference, fit a calibrator, change binning, or alter metric calculations.
This output is local/manual aggregate analysis only. It is not clinical
calibration evidence and is not deployment validation.

If a future confirmatory local run explicitly needs bootstrap intervals, set `--bootstrap-iters` to the prespecified value for that run and record it in the experiment card.

## Safety warnings

Do not commit:

- raw medical data
- private absolute paths
- model weights
- credentials or tokens
- private prediction/evaluation outputs
- ad hoc metric screenshots or spreadsheets without provenance

## Recommended local run discipline

- Use one run directory per experiment: `results/local_private_runs/<RUN_ID>/...`
- Record dataset, adapter, preprocessing, uncertainty strategy, and output locations in an experiment card
- Keep exploratory runs distinct from confirmatory runs
- Treat any change to protocol, label mapping, thresholds, or preprocessing as a new run
